import React, { useState, useEffect } from 'react';

const StatusPanel = () => {
    const [postureStatus, setPostureStatus] = useState("Unknown");
    const [eyeStatus, setEyeStatus] = useState("Unknown");
    const [blinkRate, setBlinkRate] = useState(0);
    const [fatigueLevel, setFatigueLevel] = useState(0);
    const [screenDistanceStatus, setScreenDistanceStatus] = useState("Unknown");
    const [sittingDuration, setSittingDuration] = useState(0);

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch('http://localhost:5000/status');
                const data = await response.json();
                
                // Update state with the fetched data
                setPostureStatus(data.posture_status);
                setEyeStatus(data.eye_status);
                setBlinkRate(data.blink_rate);
                setFatigueLevel(data.fatigue_level);
                setScreenDistanceStatus(data.screen_distance_status);
                setSittingDuration(data.sitting_duration);
            } catch (error) {
                console.error("Error fetching status:", error);
            }
        };

        // Fetch status every second
        const interval = setInterval(fetchStatus, 1000);

        // Cleanup interval on component unmount
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="card mt-3">
            <div className="card-body">
                <h5 className="card-title">Wellness Status</h5>
                <ul className="list-group list-group-flush">
                    <li className="list-group-item">Posture: {postureStatus}</li>
                    <li className="list-group-item">Eye Status: {eyeStatus}</li>
                    <li className="list-group-item">Blink Rate: {blinkRate.toFixed(2)} blinks/min</li>
                    <li className="list-group-item">Fatigue Level: {fatigueLevel.toFixed(2)}%</li>
                    <li className="list-group-item">Screen Distance: {screenDistanceStatus}</li>
                    <li className="list-group-item">Sitting Duration: {Math.floor(sittingDuration / 60)}m {Math.floor(sittingDuration % 60)}s</li>
                </ul>
            </div>
        </div>
    );
};

export default StatusPanel;