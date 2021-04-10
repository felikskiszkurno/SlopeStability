CREATE TABLE ExecutedTests (
    TestID INTEGER PRIMARY KEY AUTOINCREMENT ,
    DatasetID INTEGER NOT NULL,
    NormData   INTEGER,
    NormDataMethod TEXT,
    NormClass INTEGER,
    NClass INTEGER,
    Sensitivity INTEGER,
    Depth INTEGER,
    InversionID INTEGER,
    FOREIGN KEY (InversionID)
                           REFERENCES InversionParameters (InversionID),
    ResultsID INTEGER,
    'FOREIGN KEY (ResultsID)
                           REFERENCES ClassificationResults (ClassificationID)'
);