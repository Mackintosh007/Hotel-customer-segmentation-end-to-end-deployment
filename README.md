# content for README.mdHotel Customer Segmentation System
*Overview*
A machine learning system that segments hotel customers based on their booking patterns and behaviors using K-means clustering and PyTorch.

**Model Architecture**
Core Components
Algorithm: K-means Clustering

**Features Used:**

Length of stay

Booking lead time

Average daily rate (ADR)

Number of special requests

Previous cancellations

Deposit type

Customer type

**Performance Metrics**

Silhouette Score: 0.72

Number of Clusters: 4

Inertia: 156.34

**Customer Segments**
Luxury Seekers: High spending, longer stays

Business Travelers: Short stays, last-minute bookings

Leisure Travelers: Medium stays, advance bookings

Budget Conscious: Price-sensitive, minimal requests

***Technical Stack***
**Core Dependencies**
Python 3.9

PyTorch 2.3.1

Transformers 4.42.3

scikit-learn 1.4.1

FastAPI 0.111.0

Pandas 2.2.1

NumPy 1.26.4

Matplotlib 3.8.3

**Project Structure**
app.py              
cluster_interpreter.py
Dockerfile         
src/
    components/    
    pipelines/      
    utils.py        
templates/ 
    index.html         
requirements.txt    

**Setup and Execution**
*Local Development*
Create virtual environment:

python -m venv venv
.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train_model.py

Make predictions:

python src/predict.py --input data/customer_data.csv --output predictions.csv

Docker Deployment
docker build -t hotel-segmentation .
docker run -p 8000:8000 -v ${PWD}/data:/app/data hotel-segmentation

API Reference
Endpoints
GET /: Home page

POST /predict: Prediction endpoint

Sample API Request
curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "length_of_stay": 5,
        "lead_time": 30,
        "adr": 150.0,
        "special_requests": 2,
        "previous_cancellations": 0,
        "deposit_type": "No Deposit",
        "customer_type": "Transient"
      }'

**Environment Variables**
MODEL_PATH: Path to saved model (default: models/kmeans_model.pkl)

PORT: API port (default: 8000)

DEBUG: Enable debug mode (default: False)

Web Interface
Main application: http://localhost:8000

API documentation: http://localhost:8000/docs

ReDoc interface: http://localhost:8000/redoc