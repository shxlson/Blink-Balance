import React from 'react';

const FloatingWindow = ({ postureStatus, eyeStatus, blinkRate, fatigueLevel, screenDistanceStatus, sittingDuration }) => {
    return (
        <div style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '300px',
            backgroundColor: '#fff',
            border: '1px solid #ccc',
            borderRadius: '8px',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
            padding: '16px',
            zIndex: 1000
        }}>
            <h5>Wellness Status</h5>
            <ul>
                <li><strong>Posture:</strong> {postureStatus}</li>
                <li><strong>Eye Status:</strong> {eyeStatus}</li>
                <li><strong>Blink Rate:</strong> {blinkRate} blinks/min</li>
                <li><strong>Fatigue Level:</strong> {fatigueLevel}</li>
                <li><strong>Screen Distance:</strong> {screenDistanceStatus}</li>
                <li><strong>Sitting Duration:</strong> {sittingDuration} seconds</li>
            </ul>
        </div>
    );
};

export default FloatingWindow;