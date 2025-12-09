!pip install datasets chromadb bitsandbytes ipdb torch transformers torchvision gradio unsloth

# Importing all the necessary modules
from datasets import load_dataset
from PIL import Image
from typing import List, Tuple
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import chromadb
import torch._dynamo
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from unsloth import FastVisionModel
import requests
import ipdb
import gradio as gr
import os
import chromadb
import torch
from torch.utils.data import DataLoader
import numpy as np

# 


# Dataset Loading and Config
datasetClinicalQA = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
datasetDiseases = load_dataset("FreedomIntelligence/Disease_Database", "en", split="train")
datasetXray = load_dataset("hongrui/mimic_chest_xray_v_1", split = "train")

client = chromadb.Client()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = embedder.to('cuda')
device = "cuda" if torch.cuda.is_available() else "cpu"
config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    normalize_embeddings = True
)

# Models and Processors
hf_token = "Your HuggingFace Token here"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token = hf_token, quantization_config = config)
model = model.to('cuda')
clip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
clip_model.eval()
visionModel, visionTokenizer = FastVisionModel.from_pretrained(
    "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    quantization_config = config,
    use_gradient_checkpointing="unsloth",
)
FastVisionModel.for_inference(visionModel)
VitProcessor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
VitModel     = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")
VitModel.eval()
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

# Injestion to Chroma DB and conversion to embeddings
train_data = datasetClinicalQA['train']
collection = client.get_or_create_collection(name="pubmed_clinical_qa")
diseaseCollection = client.get_or_create_collection("diseases")
def injest_image_embeddings():
  batch_size      = 1000
  emb_batches     = []
  buffered_images = []

  with torch.no_grad():
      for item in datasetXray:
            img = item["image"]
            img = Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
            buffered_images.append(img)

            if len(buffered_images) == batch_size:
                inputs = clip_processor(
                    images=buffered_images,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    feats = clip_model.get_image_features(**inputs)

                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                emb_batches.append(feats.cpu())
                buffered_images.clear()

      if buffered_images:
          inputs = clip_processor(
              images=buffered_images,
              return_tensors="pt",
              padding=True
          ).to(device)

          with torch.cuda.amp.autocast(enabled=(device=="cuda")):
              feats = clip_model.get_image_features(**inputs)

          feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
          emb_batches.append(feats.cpu())
          buffered_images.clear()
  image_embs = torch.cat(emb_batches, dim=0)
  return image_embs

def injest_clinical_qa():
  batch_size = 5461
  document_batch = []
  id_batch = []
  embedding_batch = []
  for i, entry in enumerate(train_data):
    context = entry['context']['contexts']
    abstract = " ".join(context)
    doc_id = str(entry['pubid'])
    answer = entry['long_answer']
    doc = abstract + " " + answer
    document_batch.append(doc)
    id_batch.append(doc_id)
    if len(document_batch) == batch_size or i == len(train_data) - 1:
        embeddings = embedder.encode(document_batch)
        embedding_batch.extend(embeddings)
        collection.add(documents=document_batch, ids=id_batch, embeddings=embedding_batch)
        document_batch = []
        id_batch = []
        embedding_batch = []

def ingest_disease_prediction():
    batch_size = 100
    document_batch = []
    id_batch = []
    metadata_batch = []
    for i, record in enumerate(datasetDiseases):

        doc_id = str(record["disease_id"])
        disease_name = record["disease"]
        symptoms = record["common_symptom"]
        treatment = record.get("treatment", "")
        text = f"Name: {disease_name}\nSymptoms: {symptoms}\nTreatment: {treatment}"

        document_batch.append(text)
        id_batch.append(doc_id)
        metadata_batch.append({"disease": disease_name, "id": doc_id})

        if len(document_batch) == batch_size or i == len(datasetDiseases) - 1:
            embeddings = embedder.encode(document_batch, convert_to_tensor=True, normalize_embeddings=True)
            embeddings_list = embeddings.cpu().numpy().tolist()
            diseaseCollection.add(
                documents=document_batch,
                metadatas=metadata_batch,
                ids=id_batch,
                embeddings=embeddings_list
            )

            document_batch = []
            id_batch = []
            metadata_batch = []

image_embs = injest_image_embeddings()
ingest_disease_prediction()
injest_clinical_qa()

# Retrieval Functions
def retrieve_clinical_qa(query, client, collection_name="pubmed_clinical_qa", n_results=3):
    collection = client.get_collection(collection_name)
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"]

def retrieve_disease_data(symptom_text, top_k = 10):
    query_embedding = embedder.encode(symptom_text, normalize_embeddings=True)
    results = diseaseCollection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    diseases = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        diseases.append((meta["disease"], doc))
    return diseases

def retrieve_multimodal(img, k_img = 3):
    img = img.resize((512, 512))
    if img is None:
        raise ValueError("Image is not loaded properly.")
    with torch.no_grad():
        inputs    = clip_processor(images=img, return_tensors="pt").to(device)
        q_img_emb = clip_model.get_image_features(**inputs)
        q_img_emb = q_img_emb / q_img_emb.norm(p=2, dim=-1, keepdim=True)
        q_img_emb = q_img_emb.cpu().float()
    emb_corpus = image_embs.cpu().float()
    sims     = (emb_corpus @ q_img_emb.T).squeeze(1)
    top_idxs = sims.topk(k_img).indices.tolist()

    img_exs   = []
    for i in top_idxs:
        img_exs.append(datasetXray[i])
    return img_exs

def predict_cxr(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    inputs = VitProcessor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = VitModel(**inputs).logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    prob_dict = { label_columns[i]: probs[i] for i in range(len(label_columns)) }
    top_idx = int(torch.argmax(logits).item())
    return prob_dict

# Config & Pipeline for Disease Prediction
diseaseCollection.embedding_function = embed_fn

llama_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# Generators and Predictors
def generate_answer(user_notes, context, question):
    input_text = (
        "You are a senior doctor helping a nurse understand a specific patient's condition. "
        "Use the clinical notes to give a clear, medically accurate, and context-specific answer. Context is just for your medical references, it has nothing to do with patient. "
        "Avoid generalizations and irrelevant data. You must use less than 100 words.\n\n"
        f"Patient Notes (User Provided): {user_notes}\n"
        f"Retrieved Medical Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs['input_ids'].to('cuda'), max_new_tokens=200, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def clinical_qa_system(query, client, n_results=3):
    user_notes, question = query.split("Question:")
    if question == "" or question is None:
      return "Provide a question to get response!"
    retrieved_docs = retrieve_clinical_qa(query, client, n_results=n_results)
    context = ""
    context += "\n" + "\n".join([str(doc) for doc in retrieved_docs])
    answer = generate_answer(user_notes, context, question)
    answer = answer.split("Answer:")[1]
    words = answer.split()
    answer = " ".join(words[:100])
    answer = answer.split("Note")[0]
    answer = answer[:answer.rfind(".") + 1]
    return answer

def predict_diseases(symptoms):
    if symptoms == "" or symptoms is None:
      return "Please provide list of symptoms separated by comma(,).!"
    candidates = retrieve_disease_data(symptoms)
    prompt = (
        "Given the patient symptoms and disease candidates below, list the top 3 most likely diseases diagnosis with confidence levels.\n"
        f"Patient Symptoms: {symptoms}\n\n"
        "Disease Candidates:\n"
    )
    for name, info in candidates:
        prompt += f"- {name}: {info}\n"
    prompt += (
        "\nYou must only provide output as:\n"
        "1. Disease Name — Confidence (High/Medium/Low)\n"
        "2. ...\n"
        "3. ...\n"
    )
    prompt += "Answer:"
    response = llama_pipeline(prompt)[0]["generated_text"]
    response = response.split("Answer:")[1]
    lines = response.strip().split("\n")
    filtered = [line.strip() for line in lines if line.strip().startswith(("1.", "2.", "3."))]
    response = "\n".join(filtered[:3])
    return response

def predict_radiology_description(image, instruction=None):
    examples = retrieve_multimodal(image)
    text = "Similar Cases Findings:\n "
    for item in examples:
      text+=(item['text'])
    exps = "Another Model Predictions: \n"
    examplesCXR = predict_cxr(image)
    if instruction == "" or instruction is None:
      instruction = "Describe the X-Ray"
    for label, item in examplesCXR.items():
      exps+= f" {label} : {item*100} Chance"
    prompt = f"""
You are a radiologist describing **only visual findings** on a chest X-ray in **one continuous paragraph** (<= 150 words, no “\\n”, no bullets).
Avoid clinical diagnoses unless directly visible; if uncertain, prefix with “suggestive of” or “possible.”
Focus on radiographic features — pneumonia, cardiomegaly, infiltrates, consolidation, effusion, pneumothorax, etc.
**Here are some preliminary model predictions of the same scan:**
{text}
{exps}
**NOW DESCRIBE THE IMAGE BELOW IN ONE PARAGRAPH (ZERO LINE BREAKS) BASED ON INSTRUCTION:**
Instruction: {instruction} WRITE ONLY IN ONE SINGLE PARAGRAPH.
Answer:"""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    input_text = visionTokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = visionTokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    output_ids = visionModel.generate(
        **inputs,
        max_new_tokens=256,
        temperature=1.5,
        min_p=0.1
    )
    generated = visionTokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = generated.split("Note")[0]
    generated = generated.split("Answer:")[1]
    if "assistant" in generated:
      generated = generated.split("assistant")[1]
    generated = "".join(generated.splitlines())
    return f"{generated.strip()}", examplesCXR

# Launching the system
def clinical_qa_callback(user_notes, question):
    query = f"Clinical Notes: {user_notes} Question:{question}"
    return clinical_qa_system(query, client)

def disease_callback(symptoms):
    return predict_diseases(symptoms)

def radiology_callback(image, instruction):
    return predict_radiology_description(image, instruction)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Modal Clinical Assistant")

        with gr.Tab("Clinical QA"):
            gr.Markdown("### Clinical Question Answering")
            notes_input = gr.TextArea(label="Patient Notes (Clinical)")
            question_input = gr.Textbox(label="Question")
            qa_button = gr.Button("Get Answer")
            qa_output = gr.TextArea(label="Answer", lines=4)

            qa_button.click(
                clinical_qa_callback,
                inputs=[notes_input, question_input],
                outputs=qa_output
            )

        with gr.Tab("Disease Prediction"):
            gr.Markdown("### Disease Prediction from Symptoms")
            symptoms_input = gr.TextArea(label="Patient Symptoms(Enter diseases separated by comma e.g., Polyuria, unexplained weight loss, fatigue and blurred vision for better results)")
            disease_button = gr.Button("Predict Diseases")
            disease_output = gr.TextArea(label="Top 3 Diseases with Confidence", lines=4)

            disease_button.click(
                disease_callback,
                inputs=[symptoms_input],
                outputs=disease_output
            )

        with gr.Tab("Radiology Analysis"):
            gr.Markdown("### Chest X-Ray Analysis")
            image_input = gr.Image(type="pil", label="Chest X-Ray Image")
            instruction_input = gr.Textbox(label="Instruction (optional)")
            radiology_button = gr.Button("Analyze Image")
            with gr.Row():
              radiology_output = gr.TextArea(label="Radiology Description", lines=10)
              bar = gr.Label(num_top_classes=5, label="Preliminary Predictions")

            radiology_button.click(
                radiology_callback,
                inputs=[image_input, instruction_input],
                outputs=[radiology_output, bar],
                show_progress = True
            )

    demo.launch(debug = True)

if __name__ == "__main__":
    main()
