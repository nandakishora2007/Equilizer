# DeepFake Agentic AI

A lightweight hackathon MVP for detecting potentially manipulated
videos using specialized analysis agents.

## Project Structure

```text
deepfake-agentic/
│
├── app.py
├── requirements.txt
├── README.md
│
├── checkpoints/
│   └── deepfake_efficientnet_b0.pth
│
├── agents/
│   ├── __init__.py
│   ├── coordinator.py
│   ├── face_agent.py
│   ├── audio_agent.py
│   └── fusion_agent.py
│
├── models/
│   ├── __init__.py
│   └── detector.py
│
└── utils/
    ├── __init__.py
    ├── video_utils.py
    └── audio_utils.py


        Pipeline

        Video
            |
            v
    Coordinator
            |
  +---------------------+
  |                     |
  v                     v
Face Agent          Audio Agent
  |                     |
  v                     v
EfficientNet        MFCC/Spectral
Detector            Supporting Evidence
  |                     |
  +----------+----------+
             |
             v
       Fusion Agent
             |
             v
       REAL / DEEPFAKE