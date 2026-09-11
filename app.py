import streamlit as st
import torch
torch.backends.mkldnn.enabled = False
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from collections import deque

RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60

@st.cache_resource
def _get_rate_limit_store():
    """Shared across all sessions in this process, keyed by client IP.

    Streamlit re-executes this module on every rerun, so a plain module-level
    dict gets reset on each interaction. st.cache_resource persists the
    returned object across reruns instead.
    """
    return {}, threading.Lock()


def check_rate_limit():
    """Global, IP-based sliding-window rate limit. Returns (allowed, seconds_to_wait)."""
    ip = st.context.ip_address or "unknown"
    now = time.time()
    ip_request_log, lock = _get_rate_limit_store()
    with lock:
        timestamps = ip_request_log.setdefault(ip, deque())
        while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return False, RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])
        timestamps.append(now)
        return True, 0

st.set_page_config(page_title="Image Captioner", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

.main {
    background: #ffffff;
    font-family: 'Orbitron', monospace;
}
.stApp {
    background: #ffffff;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', monospace !important;
    color: #000000 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #d0d0d0;
}
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    color: #000000 !important;
    border: 1px solid #d0d0d0;
    border-radius: 5px;
    padding: 10px 20px;
    font-family: 'Orbitron', monospace;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #e0e0e0;
    color: #000000 !important;
    font-weight: bold;
}
.stButton>button {
    width: 100%;
    background: #f0f0f0;
    color: #000000;
    border-radius: 5px;
    padding: 12px;
    font-size: 16px;
    font-family: 'Orbitron', monospace;
    font-weight: bold;
    border: 1px solid #000000;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: #e0e0e0;
    transform: translateY(-2px);
}
.metric-card {
    background: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #d0d0d0;
    margin: 10px 0;
    color: #000000;
}
.metric-card h4 {
    color: #000000 !important;
}
.caption-box {
    background: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #000000;
    margin: 10px 0;
    font-size: 18px;
    color: #000000;
    font-family: 'Orbitron', monospace;
}
.stMetric {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #d0d0d0;
}
.stMetric label {
    color: #000000 !important;
    font-family: 'Orbitron', monospace !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #000000 !important;
    font-size: 2rem !important;
}
p, li, div {
    color: #000000 !important;
}
.stMarkdown {
    color: #000000;
}
[data-testid="stFileUploadDropzone"] {
    background: #f5f5f5;
    border: 2px dashed #d0d0d0;
    border-radius: 10px;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #000000;
}
[data-testid="stBaseButton-secondary"] {
    color: #ffffff !important;
}
[data-testid="stBaseButton-secondary"] * {
    color: #ffffff !important;
}
[data-testid="stFileChipName"] {
    color: #ffffff !important;
}
[data-testid="stFileChip"] span {
    color: #ffffff !important;
}
hr {
    border-color: #d0d0d0 !important;
}
[data-testid="stHeader"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

class Encoder(nn.Module):
    """Projects each of the 49 spatial ResNet50 feature vectors (2048-d) down to encoder_dim,
    instead of pooling them into a single vector."""
    def __init__(self, feature_size=2048, encoder_dim=512, dropout=0.5):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(feature_size, encoder_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, features):
        out = self.fc(features)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    """Bahdanau-style additive attention over the encoder's spatial feature grid."""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_dim, decoder_dim, attention_dim, pad_idx=0, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    def forward_step(self, encoder_out, h, c, word_idx):
        embedding = self.embedding(word_idx)
        context, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))
        gated_context = gate * context
        lstm_input = torch.cat([embedding, gated_context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        preds = self.fc(self.dropout(h))
        return preds, h, c, alpha
    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)
        h, c = self.init_hidden_state(encoder_out)
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device=encoder_out.device)
        for t in range(seq_len):
            preds, h, c, _ = self.forward_step(encoder_out, h, c, captions[:, t])
            outputs[:, t, :] = preds
        return outputs

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttentionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, image_features, captions):
        encoder_out = self.encoder(image_features)
        outputs = self.decoder(encoder_out, captions)
        return outputs

def greedy_search(model, image_features, max_length, start_token, end_token, device='cpu'):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_features)
        h, c = model.decoder.init_hidden_state(encoder_out)
        current_word = torch.tensor([start_token]).to(device)
        sequence = [start_token]
        for _ in range(max_length):
            if sequence[-1] == end_token:
                break
            preds, h, c, _ = model.decoder.forward_step(encoder_out, h, c, current_word)
            next_word = preds.argmax(dim=1).item()
            sequence.append(next_word)
            current_word = torch.tensor([next_word]).to(device)
        return sequence

def beam_search(model, image_features, max_length, start_token, end_token, idx2word, beam_width=3, device='cpu'):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_features)
        h, c = model.decoder.init_hidden_state(encoder_out)
        sequences = [[[start_token], 0.0, h, c]]
        for _ in range(max_length):
            all_candidates = []
            for seq, score, h_state, c_state in sequences:
                if seq[-1] == end_token:
                    all_candidates.append([seq, score, h_state, c_state])
                    continue
                current_word = torch.tensor([seq[-1]]).to(device)
                preds, new_h, new_c, _ = model.decoder.forward_step(encoder_out, h_state, c_state, current_word)
                probs = torch.log_softmax(preds[0], dim=0)
                top_probs, top_indices = probs.topk(beam_width)
                for i in range(beam_width):
                    candidate = [seq + [top_indices[i].item()], score + top_probs[i].item(), new_h, new_c]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
        return sequences[0][0]

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
        vocab = vocab_data['vocab']
        max_length = vocab_data['max_length']
        vocab_size = len(vocab)
        encoder = Encoder(2048, 512)
        decoder = DecoderWithAttention(vocab_size, 256, 512, 512, 256, pad_idx=0)
        model = Seq2SeqAttentionModel(encoder, decoder)
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        model.eval()
        return model, word2idx, idx2word, max_length, device
    except:
        return None, None, None, None, None

def generate_caption(image, model, word2idx, idx2word, max_length, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-2]).to(device)
    resnet.eval()
    with torch.no_grad():
        feats = resnet(img_tensor)
        features = feats.permute(0, 2, 3, 1).reshape(feats.size(0), -1, 2048)

    def to_words(indices):
        return ' '.join([idx2word[idx] for idx in indices if idx not in [0, word2idx['<start>'], word2idx['<end>']]])

    greedy_indices = greedy_search(model, features, max_length, word2idx['<start>'], word2idx['<end>'], device=device)
    beam_indices = beam_search(model, features, max_length, word2idx['<start>'], word2idx['<end>'], idx2word, beam_width=5, device=device)
    return to_words(greedy_indices), to_words(beam_indices)

st.title("Image Captioner")

model, word2idx, idx2word, max_length, device = load_model()

if model is None:
    st.error("MODEL NOT FOUND // Please ensure 'model_weights.pth' and 'vocab.pkl' are in the same directory.")
else:
    tab1, tab2, tab3 = st.tabs(["GENERATE CAPTION", "MODEL PERFORMANCE", "ABOUT"])
    
    with tab1:
        ALLOWED_FORMATS = ['JPEG', 'PNG']
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload an image", label_visibility="collapsed")
            is_valid_image = False
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    detected_format = image.format
                except Exception:
                    detected_format = None

                if detected_format not in ALLOWED_FORMATS:
                    label = f".{detected_format.lower()}" if detected_format else "This"
                    st.error(f"Please upload a JPG or PNG image.")
                else:
                    is_valid_image = True
                    image = image.convert('RGB')
                    st.image(image, caption="UPLOADED IMAGE", width='stretch')
        with col2:
            if uploaded_file and is_valid_image:
                file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
                if st.session_state.get("cached_file_id") == file_id:
                    greedy_caption, beam_caption = st.session_state.cached_captions
                else:
                    allowed, wait_seconds = check_rate_limit()
                    if not allowed:
                        st.error(
                            f"Can only upload {RATE_LIMIT_MAX_REQUESTS} images per "
                            f"{RATE_LIMIT_WINDOW_SECONDS} seconds. Try again in {wait_seconds:.0f}s."
                        )
                        st.stop()
                    with st.spinner("Generating caption..."):
                        greedy_caption, beam_caption = generate_caption(image, model, word2idx, idx2word, max_length, device)
                    st.session_state.cached_file_id = file_id
                    st.session_state.cached_captions = (greedy_caption, beam_caption)
                st.markdown(f'<div class="caption-box"><b>GREEDY CAPTION:</b><br><br>{greedy_caption}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="caption-box"><b>BEAM SEARCH CAPTION:</b><br><br>{beam_caption}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### TRAINING RESULTS")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BLEU-4 (Greedy)", "0.3087")
        with col2:
            st.metric("BLEU-4 (Beam)", "0.3793")
        with col3:
            st.metric("Vocabulary Size", f"{len(idx2word)}", help="Total unique words in vocabulary")

        st.markdown("### MODEL ARCHITECTURE")
        st.markdown("""
        <div class="metric-card">
        <ul style="margin: 0; padding-left: 20px; color: #000000;">
            <li><b style="color: #000000;">Encoder:</b> ResNet50 (spatial 7×7 grid) → Linear(2048 → 512) per location</li>
            <li><b style="color: #000000;">Attention:</b> Bahdanau-style additive attention over the 49 spatial locations</li>
            <li><b style="color: #000000;">Decoder:</b> Embedding(256) → LSTMCell(512) with attention context → Linear(vocab_size=7689)</li>
            <li><b style="color: #000000;">Training:</b> 28 epochs, Adam optimizer, CrossEntropyLoss</li>
            <li><b style="color: #000000;">Dataset:</b> Flickr30k (31,783 images, 158,915 captions)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### PROJECT OVERVIEW")
        st.markdown("""
        <div class="metric-card">
       
        <p style="color: #000000 !important;">This project implements an attention Sequence-to-Sequence (Seq2Seq) model for automatic image captioning using PyTorch.</p>
        
        <h4 style="color: #000000 !important;">Key Features:</h4>
        <ul style="color: #000000 !important;">
            <li>Pre-trained ResNet50 for image feature extraction</li>
            <li>LSTM-based decoder for caption generation</li>
            <li>Beam search for improved caption quality</li>
            <li>Trained on Flickr30k dataset</li>
        </ul>
        
        <h4 style="color: #000000 !important;">Technologies Used:</h4>
        <ul style="color: #000000 !important;">
            <li>PyTorch, Torchvision</li>
            <li>NLTK for text processing</li>
            <li>Streamlit for web interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
