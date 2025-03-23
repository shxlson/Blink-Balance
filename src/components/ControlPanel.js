import React from 'react';
import axios from 'axios';

const ControlPanel = ({ onStart, onStop }) => {
    const handleStart = async () => {
        try {
            await axios.post('http://localhost:5000/start');
            onStart();
        } catch (error) {
            console.error('Error starting WellnessGuard:', error);
        }
    };

    const handleStop = async () => {
        try {
            await axios.post('http://localhost:5000/stop');
            onStop();
        } catch (error) {
            console.error('Error stopping WellnessGuard:', error);
        }
    };

    return (
        <div className="card mt-3">
            <div className="card-body">
                <h5 className="card-title">Blink Balance wants to access your webcam-DISCLAIMER-Your data is not being stored respecting your security</h5>
                <button className="btn btn-success me-2" onClick={handleStart}>Allow</button>
                <button className="btn btn-danger" onClick={handleStop}>Block</button>
            </div>
        </div>
    );
};

export default ControlPanel;