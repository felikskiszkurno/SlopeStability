CREATE TABLE ClassificationResults (

    ClassificationID INTEGER PRIMARY KEY AUTOINCREMENT,

    TrainingADA NUMERIC,
    TrainingGBC NUMERIC,
    TrainingKNN NUMERIC,
    TrainingSGD NUMERIC,
    TrainingSVM NUMERIC,

    PredictionADA NUMERIC,
    PredictionGBC NUMERIC,
    PredictionKNN NUMERIC,
    PredictionSGD NUMERIC,
    PredictionSVM NUMERIC

);