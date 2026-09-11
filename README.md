# Image Captioning AI

An image captioning AI model using attention based Seq2Seq architecture.
Interactive UI available at: https://image-captioner-znx9.onrender.com/

## Model Details

- **Architecture**: ResNet50 (spatial features) Encoder + Attention + LSTM Decoder
- **Dataset**: Flickr30k (31,783 images)
- **BLEU-4 Score**: 0.3793 (Beam Search), 0.3087(Greedy search)
- **Vocabulary**: 7,689 words

## Site Features

- Real-time AI caption generation
- Model performance metrics

## Usage

1. Navigate to the "Generate Caption" tab
2. Upload an image (JPEG or PNG)
3. View the generated caption

## Local Setup

1. From root folder, run:

```bash
docker compose up -d --build
```

2. Open http://localhost:8501

## Project Structure

```
.
├── app.py                          # Streamlit application
├── model.ipynb                     # model and training notebook
├── requirements.txt                # Python dependencies (app)
├── requirements-model.txt          # Python dependencies (training notebook)
├── model_weights.pth               # Trained model weights
├── vocab.pkl                       # Vocabulary
├── Dockerfile                      # dockerfile for image build
├── docker-compose.yml              # Docker Compose file

```
