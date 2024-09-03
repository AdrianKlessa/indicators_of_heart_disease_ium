export KAGGLE_USERNAME=${kaggle_username}
export KAGGLE_KEY=${kaggle_password}
kaggle datasets download -d kamilpytlak/personal-key-indicators-of-heart-disease/
python ./data_cleanup.py