import React from 'react';

const AlertPanel = ({ alerts }) => {
    return (
        <div className="card mt-3">
            <div className="card-body">
                <h5 className="card-title">Alerts</h5>
                {alerts.length > 0 ? (
                    <ul className="list-group list-group-flush">
                        {alerts.map((alert, index) => (
                            <li key={index} className="list-group-item">
                                <strong>{alert.type}:</strong> {alert.message}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p>No alerts at the moment.</p>
                )}
            </div>
        </div>
    );
};

export default AlertPanel;