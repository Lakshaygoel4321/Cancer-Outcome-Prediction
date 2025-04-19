import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const initialState = {
    Age: '',
    Gender: '',
    Nationality: '',
    Emirate: '',
    Cancer_Type: '',
    Cancer_Stage: '',
    Treatment_Type: '',
    Hospital: '',
    Smoking_Status: '',
    Ethnicity: '',
    Height: '',
    Weight: '',
  };

  const [formData, setFormData] = useState(initialState);
  const [prediction, setPrediction] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({...formData, [name]: value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(res.data.prediction);
    } catch (err) {
      console.error('Prediction failed:', err);
    }
  };

  const renderInput = (label, name, type = 'text') => (
    <div className="form-group">
      <label>{label}</label>
      <input type={type} name={name} value={formData[name]} onChange={handleChange} required />
    </div>
  );

  const renderSelect = (label, name, options) => (
    <div className="form-group">
      <label>{label}</label>
      <select name={name} value={formData[name]} onChange={handleChange} required>
        <option value="">Select {label}</option>
        {options.map(opt => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="App">
      <h1>Cancer Outcome Predictor</h1>
      <form onSubmit={handleSubmit}>
        {renderInput("Age", "Age", "number")}
        {renderSelect("Gender", "Gender", ["Male", "Female"])}
        {renderSelect("Nationality", "Nationality", ["Emirati", "Expatriate"])}
        {renderSelect("Emirate", "Emirate", ["Umm Al Quwain", "Abu Dhabi", "Fujairah", "Ras Al Khaimah", "Sharjah", "Dubai", "Ajman"])}
        {renderSelect("Cancer Type", "Cancer_Type", ["Liver", "Leukemia", "Lung", "Pancreatic", "Breast", "Ovarian", "Prostate", "Colorectal"])}
        {renderSelect("Cancer Stage", "Cancer_Stage", ["I", "II", "III", "IV"])}
        {renderSelect("Treatment Type", "Treatment_Type", ["Radiation", "Surgery", "Chemotherapy", "Immunotherapy"])}
        {renderSelect("Hospital", "Hospital", ["Sheikh Khalifa Hospital", "Dubai Hospital", "Zayed Military Hospital", "Cleveland Clinic Abu Dhabi"])}
        {renderSelect("Smoking Status", "Smoking_Status", ["Non-Smoker", "Smoker", "Former Smoker"])}
        {renderSelect("Ethnicity", "Ethnicity", ["European", "South Asian", "African", "East Asian", "Arab"])}
        {renderInput("Height (cm)", "Height", "number")}
        {renderInput("Weight (kg)", "Weight", "number")}
        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <div className="result">
          <h2>Predicted Outcome: <span>{prediction}</span></h2>
        </div>
      )}
    </div>
  );
}

export default App;
