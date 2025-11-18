import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // State for the uploaded file
  const [selectedFile, setSelectedFile] = useState(null);
  // State for the image preview URL
  const [preview, setPreview] = useState(null);
  // State for the API response
  const [result, setResult] = useState(null);
  // State for loading and errors
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 1. Handles file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null); // Clear previous results
      setError(null); // Clear previous errors
      
      // Create a preview URL for the selected image
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // 2. Handles the "Analyze" button click
  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Call the FastAPI backend (must be running!)
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Store the result
      setResult(response.data);
      
    } catch (err) {
      console.error(err);
      if (err.response) {
        setError(`Error: ${err.response.data.detail || 'Failed to get response'}`);
      } else {
        setError("Network Error: Is the backend server running?");
      }
    } finally {
      setLoading(false);
    }
  };

  // Helper to determine risk color
  const getRiskColor = (diagnosis) => {
    if (diagnosis === 'mel' || diagnosis === 'bcc' || diagnosis === 'akiec') {
      return 'risk-high'; // Malignant/Pre-malignant
    }
    if (diagnosis === 'bkl' || diagnosis === 'df' || diagnosis === 'vasc') {
      return 'risk-medium'; // Benign, but notable
    }
    return 'risk-low'; // Benign (nv)
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🩺 DermaNet-X: Explainable AI</h1>
        <p>Upload a skin lesion image for analysis and explanation.</p>
      </header>

      <main className="content">
        <div className="upload-section">
          <input type="file" onChange={handleFileChange} accept="image/png, image/jpeg" />
          <button onClick={handleSubmit} disabled={loading || !selectedFile}>
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="results-container">
          {/* Left Panel: Original Image Preview */}
          {preview && (
            <div className="image-card">
              <h3>Original Image</h3>
              <img src={preview} alt="Uploaded lesion preview" />
            </div>
          )}

          {/* Right Panel: AI Results & Heatmap */}
          {result && (
            <div className="image-card">
              <h3>AI Analysis (Grad-CAM)</h3>
              <img src={result.heatmap_base64} alt="AI heatmap explanation" />
            </div>
          )}
        </div>

        {/* Results Info Box */}
        {result && (
          <div className="info-card">
            <h2>Analysis Complete</h2>
            <div className="info-grid">
              <strong>Diagnosis:</strong>
              <span className={`diagnosis-pill ${getRiskColor(result.diagnosis)}`}>
                {result.diagnosis.toUpperCase()}
              </span>

              <strong>Confidence:</strong>
              <span>{result.confidence}%</span>
              
              <strong>Uncertainty:</strong>
              <span className={result.risk_flag === 'High' ? 'risk-high' : 'risk-low'}>
                {result.risk_flag} ({result.uncertainty_score.toFixed(4)})
              </span>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;