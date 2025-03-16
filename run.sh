echo "Downloading agri.db..."
curl -L -o src/data/agri.db https://techassessment.blob.core.windows.net/aiip5-assessment-data/agri.db
echo "Starting model training..."
python src/main.py
echo "Pipeline execution complete!"