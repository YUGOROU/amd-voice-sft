import { useState } from 'react';
import { Menu, X, Mic, MicOff, PhoneOff, Calendar, Bell, Edit3 } from 'lucide-react';
import './index.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [micActive, setMicActive] = useState(false);
  const [notepadText, setNotepadText] = useState("• Ask doctor about new medication\n• Call Sarah this weekend");

  const [reminders, setReminders] = useState([
    { id: 1, text: "Drink Water", done: false },
    { id: 2, text: "Morning Medicine", done: true },
    { id: 3, text: "Lunch", done: false },
    { id: 4, text: "Evening Medicine", done: false },
  ]);

  const toggleReminder = (id: number) => {
    setReminders(reminders.map(r => r.id === id ? { ...r, done: !r.done } : r));
  };

  return (
    <div className="app-container">
      {/* Top Bar for Sidebar Toggle */}
      {!sidebarOpen && (
        <div className="top-bar">
          <button className="toggle-sidebar-btn" onClick={() => setSidebarOpen(true)}>
            <Menu size={24} />
          </button>
        </div>
      )}

      {/* Expandable Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2>Lumi Dashboard</h2>
          <button className="control-btn" style={{ width: 40, height: 40 }} onClick={() => setSidebarOpen(false)}>
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-content">
          {/* Notepad Widget */}
          <div className="widget">
            <h3><Edit3 size={16} /> Notepad</h3>
            <textarea 
              value={notepadText}
              onChange={(e) => setNotepadText(e.target.value)}
              placeholder="Jot down your thoughts..."
            />
          </div>

          {/* Reminders Widget */}
          <div className="widget">
            <h3><Bell size={16} /> Daily Reminders</h3>
            <div className="reminders-list">
              {reminders.map(reminder => (
                <label key={reminder.id} className="reminder-item" style={{ opacity: reminder.done ? 0.6 : 1 }}>
                  <input 
                    type="checkbox" 
                    checked={reminder.done} 
                    onChange={() => toggleReminder(reminder.id)} 
                  />
                  <span style={{ textDecoration: reminder.done ? 'line-through' : 'none' }}>
                    {reminder.text}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Appointments Widget */}
          <div className="widget">
            <h3><Calendar size={16} /> Appointments</h3>
            <div className="appointments-list">
              <div className="appointment">
                <div className="apt-time">09:00 AM</div>
                <div className="apt-details">Doctor's Appointment</div>
              </div>
              <div className="appointment">
                <div className="apt-time">01:30 PM</div>
                <div className="apt-details">Call Daughter</div>
              </div>
              <div className="appointment">
                <div className="apt-time">04:00 PM</div>
                <div className="apt-details">Walk in the park</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Video Area (Google Meet Style) */}
      <div className={`main-area ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="video-container">
          {/* Placeholder for the SadTalker Avatar Video stream */}
          {/* Using a placeholder portrait for Dorothy (Elderly Black woman) as an example */}
          <img 
            src="https://i.pravatar.cc/1000?img=5" 
            alt="Lumi Avatar" 
            className="avatar-image" 
          />

          {/* Google Meet style bottom controls */}
          <div className="controls-overlay">
            <button 
              className={`control-btn ${micActive ? 'mic-on' : ''}`}
              onClick={() => setMicActive(!micActive)}
            >
              {micActive ? <Mic size={24} color="#202124" /> : <MicOff size={24} />}
            </button>
            <button className="control-btn danger">
              <PhoneOff size={24} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
