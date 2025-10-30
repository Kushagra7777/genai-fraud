# AI Video Detector



### 1. Create a Python Virtual Environment
```bash
python -m venv myenv
```
### 2. activate env
```bash
myenv/scripts/activate
```


### 3. install dependencies
```bash
pip install opencv-python openai python-dotenv
```

### 4. create .env file with below context
```bash
OPENAI_API_KEY=sk-proj-......
```



### 5. Run the script with this command
```bash
python script.py --video videos/Real/video-662.mp4 --out video_analysis.json --save-frames frames_out --interval 0.5 --max-total 60 --batch-size 20 --model gpt-4o
```