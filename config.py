class Config:
    CSV_PATH = 'data_test.csv'
    DATAFRAME_PATH = "prepped_dataframe.pkl"
    POSTERS_PATH = 'posters/'
    MODEL_PATH = 'checkpoints/cp.ckpt'
    CSV_COLUMNS = ['id', 'title', 'year', 'score', 'genres', 'image']
    DROP_COLUMNS = ['id', 'title', 'year', 'score', 'genres', 'image']
    POSTER_WIDTH = 182
    POSTER_HEIGHT = 268
    INPUT_SHAPE = (182, 268, 3)
