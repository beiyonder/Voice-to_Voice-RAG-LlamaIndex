# voice-to-voice rag with llamaindex (rocm, on amd mi300x gpus)

high-level summary
this repository contains a jupyter nbk that demonstrates a local end-to-end voice assistant pipeline running on amd's mi300x gpus (rocm). the pipeline:
- transcribes audio -> text (whisper),
- augments the query with retrieval (llamaindex + vectorstoreindex),
- generates text responses with ollama (llama3),
- converts response text back to speech (chattts + torchaudio).

environment & supported hardware
- os: ubuntu 22.04
- amd gpus tested: instinct mi300x, radeon w7900 (rocm-compatible)
- rocm: 6.2.0
- python: 3.10
- pytorch: 2.3.0 (rocm build)

quick table of contents
1. high-level summary
2. prerequisites
3. prepare the inference environment
4. run the notebook — minimal steps
5. pipeline overview
6. theory & connections (personal notes)
7. key files / variables
8. troubleshooting

prerequisites
- rocm installed and configured for your amd gpu (rocm 6.2.0 recommended).
- conda (miniconda/anaconda).
- system packages: curl (for ollama install).
- ensure sufficient disk space for models and embeddings.

prepare the inference environment
1. create and activate conda env:
   conda create -n rocm python=3.10
   conda activate rocm

2. install rocm pytorch:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

3. install ollama and start server:
   sudo apt install curl -y
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull llama3

4. install python dependencies:
   pip install \
     llama-index \
     llama-index-llms-ollama \
     llama-index-embeddings-ollama \
     llama-index-embeddings-huggingface \
     openai-whisper \
     transformers \
     ChatTTS \
     torchaudio \
     jupyter

run the notebook — minimal steps
1. place your audio file (e.g., summarize_question.wav) in the notebook working directory.
2. ensure a `data/` directory exists with text files for retrieval. if empty, download sample:
   mkdir -p data && curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/pg11.txt
3. start jupyter:
   jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
4. open the notebook and run cells sequentially:
   - set optional rocm env vars
   - verify torch and gpu
   - download or load audio
   - transcribe with whisper
   - build llamaindex vectors from `data/`
   - query with llama3 (ollama)
   - convert response to audio using chattts -> save with torchaudio
   - play saved audio

pipeline overview (short)
- stt: whisper model (load_model("base")) transcribes AUDIO_FILE -> input_text
- documents: SimpleDirectoryReader loads files from DATA_DIR -> documents
- embeddings: HuggingFaceEmbedding (BAAI/bge-base-en-v1.5) configured via Settings.embed_model
- llm: Ollama llama-3 via Settings.llm, server must be running locally
- index & query: VectorStoreIndex.from_documents(...) -> index.as_query_engine(...) -> query(input_text)
- tts: ChatTTS.Chat().load() -> chat.infer(...) -> save waveform with torchaudio.save

theory & connections (personal notes)
- why rag here: it keeps the llm focused and grounded in a changing knowledge base. whisper gives us raw user intent, but without relevant context the model can hallucinate; attaching a vector index lets us surface relevant passages before generation.
- embeddings -> similarity search: embeddings (bge-base) map text to a dense space where semantic neighbors live close together. that means a short user request can pull back long documents that are semantically related even if keywords differ.
- vector store and query engine: think of the index as a fast filter. the query engine picks top-k candidates, and those chunks form the "context" for the llm. choosing chunk size and k trades off latency and relevance.
- llm role (ollama / llama3): ollama hosts the model locally so inference keeps data on-prem. the llm consumes the retrieved context and produces a response; streaming responses are useful to present partial results and detect early failure modes.
- tts considerations: chattts converts text to waveform; quality vs. speed depends on model compile flags and whether you infer on gpu. saving with torchaudio is straightforward, but be mindful of sample rates and channel format.
- resource notes: on mi300x the memory bandwidth and matrix kernels differ from nvidia hardware. rocm builds of pytorch can behave slightly differently; always verify torch.cuda.is_available() and perform small warmup runs to catch kernel mismatches.
- failure modes: empty data directories, misconfigured ollama server, or corrupted audio files are the most common. defensively check file existence and catch exceptions around model loading.

key files / variables
- AUDIO_FILE = "summarize_question.wav"
- DATA_DIR = "./data" (must contain text files)
- OUTPUT_AUDIO_FILE = "voice_pipeline_response.wav"
- SAMPLE_RATE = 24000
- env vars you may set:
  - TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (optional)
  - HIP_VISIBLE_DEVICES="0" (select gpu)

minimal command examples used in the notebook
- download sample audio:
  curl -L https://raw.githubusercontent.com/ROCm/gpuaidev/main/docs/notebooks/assets/summarize_question.wav -o summarize_question.wav
- download sample text:
  mkdir -p data && curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/pg11.txt

troubleshooting
- if pytorch does not detect gpu:
  - confirm rocm is installed and kernel modules loaded.
  - verify `torch.__version__` matches the rocm wheel installed.
  - check HIP_VISIBLE_DEVICES and permissions.
- ollama errors:
  - ensure `ollama serve` is running and reachable.
  - confirm `ollama pull llama3` completed successfully.
- empty data directory:
  - add text files to `data/` or run the optional download step.
- tts errors:
  - ChatTTS requires model files and cpu/gpu memory; try chat.load(compile=False) for faster startup.

what did I build?
a local, rocm-enabled voice assistant proof-of-concept that demonstrates integration of speech-to-text, retrieval-augmented generation, and text-to-speech using llamaindex + ollama on amd gpus.



