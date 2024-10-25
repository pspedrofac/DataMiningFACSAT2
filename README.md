
# Microservices Architecture for Satellite Preventive Maintenance Using Machine Learning (Facsat2)

## Overview
The **Facsat2** project is focused on implementing a machine learning-based system for **preventive maintenance of satellites**. Specifically, this architecture is designed to detect faults in the Facsat2 satellite, enhancing its reliability and performance in space by analyzing telemetry data and identifying early signs of failure.

The system leverages a **microservices architecture**, allowing for high scalability, modularity, and ease of deployment. The components interact through REST APIs, manage machine learning experiments, orchestrate tasks, and store artifacts and telemetry data.

## Architecture Components

### 1. **Telemetry Data Interface**
- **Purpose**: Serves as the input point for satellite telemetry data, allowing operators to upload and analyze satellite health information. The interface interacts with the system to detect potential faults and provide predictions for preventive maintenance.
- **Connection**: Sends telemetry data to the REST API for further analysis and fault detection.

### 2. **REST API (FastAPI)**
- **Purpose**: The core service responsible for managing communication between the system components. It processes telemetry data, interacts with machine learning models for fault detection, and returns results to the interface.
- **Connection**: Interfaces with MLflow for managing models, retrieves metadata from PostgreSQL, and sends tasks to Apache Airflow for orchestration.

### 3. **MLflow**
- **Purpose**: Manages machine learning experiments and models. It tracks the performance of the models used for fault detection and manages the storage of artifacts, including telemetry data and model parameters.
- **Connection**: Works with MinIO to store artifacts, interacts with the REST API to deploy and update models, and stores metadata in PostgreSQL.

### 4. **Apache Airflow**
- **Purpose**: Orchestrates workflows related to satellite telemetry analysis and machine learning model management. It schedules tasks such as data ingestion, preprocessing, and fault detection analysis.
- **Connection**: Collaborates with MLflow to schedule the execution of machine learning models and ensure that workflows are completed efficiently. It also interacts with PostgreSQL for storing task execution metadata.

### 5. **MinIO**
- **Purpose**: An S3-compatible object storage system used to store machine learning artifacts such as telemetry data and models. It provides persistent storage for the system's datasets and model outputs.
- **Connection**: Receives and stores artifacts from MLflow, enabling access to historical data and models used for satellite fault detection.

### 6. **PostgreSQL**
- **Purpose**: Serves as the primary relational database for storing metadata, including experiment logs, model performance metrics, and workflow execution data.
- **Connection**: Supports both MLflow and Apache Airflow by storing and managing operational data related to satellite fault detection tasks.

### 7. **MongoDB**
- **Purpose**: Stores unstructured or semi-structured satellite telemetry data for further analysis and integration with machine learning workflows.
- **Connection**: Provides persistent storage for telemetry data ingested from satellite systems and utilized by the REST API and machine learning models.
- **Ports**: Exposes the default MongoDB port `27017` and connects to other services in the system. Accessible at:
  - MongoDB port: `27017`

---

## Setting Up the System

### 1. **Install Docker & Docker Compose**

Before you begin, ensure Docker and Docker Compose are installed. If not, follow these steps:

```bash
sudo apt update -y
sudo apt install -y docker-ce docker-compose
```

### 2. **Clone the Repository and Build the Architecture**

Clone the project repository and build the Docker containers:

```bash
git clone https://github.com/your-repo/facsat2-fault-detection.git
cd facsat2-fault-detection
docker-compose build
docker-compose up -d
```

### 3. **Database Configuration**

Ensure that the `mlflow_db` database exists for MLflow by running:

```bash
docker exec -it postgres_airflow psql -U airflow
CREATE DATABASE mlflow_db;
```

### 4. **Access the Services**

Once the services are running, access the different components locally:

- **Apache Airflow**: [http://localhost:8080](http://localhost:8080)
- **MLflow**: [http://localhost:5000](http://localhost:5000)
- **MinIO (Service)**: [http://localhost:9000](http://localhost:9000)
- **MinIO (Buckets Administration)**: [http://localhost:9001](http://localhost:9001)
- **Telemetry Data Interface**: [http://localhost:8800](http://localhost:8800)
- **FastAPI**: [http://localhost:8000](http://localhost:8000)
- **MongoDB**: MongoDB runs at `mongodb://localhost:27017`

If you're using an external server, replace `localhost` with the server's IP address.

### 5. **Configure `.env` File**

Modify the `.env` file to customize the ports, database credentials, and other environment variables.

---

## Architecture Diagram

This architecture relies on Docker-based microservices that communicate through well-defined APIs. Each service performs a specialized function, whether it be managing data, running machine learning models, or orchestrating tasks. The diagram below outlines the flow of data between services, ensuring efficient detection and management of potential faults in the satellite.

---

## Future Improvements

- **Scalability**: Use Kubernetes to deploy and manage the system at scale.
- **Security**: Implement OAuth or API gateways to secure access to the system components.
- **Monitoring**: Integrate Prometheus and Grafana for real-time system monitoring and health checks.
- **Fault Tolerance**: Ensure failover mechanisms are in place to handle component failures without system-wide disruptions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
