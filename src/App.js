import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import StatusPanel from './components/StatusPanel';
import AlertPanel from './components/AlertPanel';
import ControlPanel from './components/ControlPanel';
import 'bootstrap/dist/css/bootstrap.min.css';

const App = () => {
    const [status, setStatus] = useState({
        postureStatus: "Unknown",
        eyeStatus: "Unknown",
        blinkRate: 0,
        fatigueLevel: 0,
        screenDistanceStatus: "Unknown",
        sittingDuration: 0,
        alerts: []
    });

    const [videoSrc, setVideoSrc] = useState(null); // Initialize to null instead of an empty string

    const fetchStatus = async () => {
        try {
            const response = await axios.get('http://localhost:5000/status');
            setStatus(response.data);
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    };

    useEffect(() => {
        const interval = setInterval(fetchStatus, 1000); // Fetch status every second
        setVideoSrc('http://localhost:5000/video_feed'); // Set webcam feed URL
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="container">
            <Navbar />
            <ControlPanel onStart={fetchStatus} onStop={fetchStatus} />
            <div className="row mt-3">
                <div className="col-md-6">
                    <StatusPanel {...status} />
                    <AlertPanel alerts={status.alerts} />
                </div>
                <div className="col-md-6">
                    {videoSrc && ( // Only render the img element if videoSrc is not null
                        <img src={videoSrc} alt="Webcam Feed" style={{ width: '100%', height: 'auto' }} />
                    )}
                </div>
            </div>
        </div>
    );
};

export default App;