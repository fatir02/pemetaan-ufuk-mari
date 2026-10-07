import os
import sys

app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pemetaan-ufuk-mari")
sys.path.insert(0, app_dir)
os.chdir(app_dir)

if __name__ == "__main__":
    from main import main
    main()
