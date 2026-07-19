@echo off
cd /d "C:\Users\czy66\Desktop\dwl_reproduce2"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo [%date% %time%] Training started (v2 - Critic fixed) > train_log.txt
"C:\Users\czy66\Desktop\dwl_reproduce\venv310\Scripts\python.exe" -u train.py --num_iterations 3000 --run_name dwl_run7 >> train_log.txt 2>&1
echo [%date% %time%] Training finished >> train_log.txt
