import { useState, useEffect } from 'react';
import { 
  Menu, X, Mic, MicOff, PhoneOff, Phone, Calendar as CalendarIcon, 
  Bell, Edit3, MapPin, AlertTriangle, Clock, Send, ChevronDown, Plus, Trash2, ChevronLeft, Sun, Moon
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const AVATARS = [
  { id: 'dorothy', name: 'Dorothy (Elderly Black Woman)', img: '/avatars/dorothy.png' },
  { id: 'harold', name: 'Harold (Elderly White Man)', img: '/avatars/harold.png' },
  { id: 'marcus', name: 'Marcus (Young Black Man)', img: '/avatars/marcus.png' },
  { id: 'priya', name: 'Priya (Young South Asian Woman)', img: '/avatars/priya.png' },
  { id: 'carlos', name: 'Carlos (Middle-aged Latino Man)', img: '/avatars/carlos.png' }
];

export default function Companion() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState('notepad');
  const [callActive, setCallActive] = useState(true);
  const [micActive, setMicActive] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<{role: string, text: string}[]>([]);
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0]);
  const [showAvatarMenu, setShowAvatarMenu] = useState(false);

  // Theme State
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [isDarkMode]);

  // Notes State
  const [notes, setNotes] = useState([
    { id: 1, date: '2026-05-08', content: 'Ask doctor about new medication.\nAlso check blood pressure.' },
    { id: 2, date: '2026-05-09', content: 'Grocery list:\n- Milk\n- Eggs\n- Bread' }
  ]);
  const [activeNoteId, setActiveNoteId] = useState<number | null>(null);

  // Calendar State
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [appointments, setAppointments] = useState<Record<string, {time: string, title: string}[]>>({
    '2026-05-09': [{time: '09:00 AM', title: 'Dr. Smith Appointment'}, {time: '01:30 PM', title: 'Call Daughter'}],
    '2026-05-12': [{time: '11:00 AM', title: 'Physiotherapy'}]
  });
  const [newApptTime, setNewApptTime] = useState("");
  const [newApptTitle, setNewApptTitle] = useState("");

  // Reminders State
  const [reminders, setReminders] = useState([
    { id: 1, text: "Morning Medicine", done: true },
    { id: 2, text: "Drink Water", done: false }
  ]);
  const [newReminder, setNewReminder] = useState("");

  // Alarms State
  const [alarms, setAlarms] = useState([
    { id: 1, time: "07:00", active: true },
    { id: 2, time: "14:30", active: false }
  ]);
  const [newAlarmTime, setNewAlarmTime] = useState("");

  // SOS State
  const [guardianNumber, setGuardianNumber] = useState("");
  const [locationShared, setLocationShared] = useState(false);
  const [sosActive, setSosActive] = useState(false);

  // Alarm Effect
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      const currentTimeString = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      alarms.forEach(alarm => {
        if (alarm.active && alarm.time === currentTimeString && now.getSeconds() === 0) {
          const audio = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3');
          audio.play().catch(e => console.log("Audio play blocked by browser", e));
        }
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [alarms]);

  const handleSendMessage = () => {
    if(!chatInput.trim()) return;
    setMessages([...messages, { role: 'user', text: chatInput }]);
    setChatInput('');
    setTimeout(() => {
      setMessages(prev => [...prev, { role: 'ai', text: "I've noted that down for you. How else can I help?" }]);
    }, 1000);
  };

  // --- Render Functions for Sidebar Tabs --- //

  const renderNotepad = () => {
    if (activeNoteId !== null) {
      const note = notes.find(n => n.id === activeNoteId)!;
      return (
        <div className="tab-container flex-col h-full">
          <div className="flex justify-between items-center mb-4">
            <button className="back-btn mb-0" onClick={() => setActiveNoteId(null)}>
              <ChevronLeft size={16} /> Back
            </button>
            <button className="del-btn text-danger flex items-center gap-1" onClick={() => {
              setNotes(notes.filter(n => n.id !== activeNoteId));
              setActiveNoteId(null);
            }}>
              <Trash2 size={16} /> Delete
            </button>
          </div>
          <div className="note-date">{note.date}</div>
          <textarea 
            className="full-textarea flex-1 mt-4" 
            value={note.content}
            onChange={(e) => {
              setNotes(notes.map(n => n.id === activeNoteId ? {...n, content: e.target.value} : n));
            }}
          />
        </div>
      );
    }
    
    return (
      <div className="tab-container">
        <button className="utility-btn w-full mb-4 flex-center" onClick={() => {
          const newNote = { id: Date.now(), date: new Date().toISOString().split('T')[0], content: '' };
          setNotes([newNote, ...notes]);
          setActiveNoteId(newNote.id);
        }}>
          <Plus size={16} /> New Note
        </button>
        <div className="history-list">
          {notes.map(note => (
            <div key={note.id} className="history-card" onClick={() => setActiveNoteId(note.id)}>
              <div className="history-date">{note.date}</div>
              <div className="history-preview">{note.content.substring(0, 40) || "Empty note..."}</div>
            </div>
          ))}
          {notes.length === 0 && <p className="text-muted text-sm text-center mt-8">No notes yet.</p>}
        </div>
      </div>
    );
  };

  const renderCalendar = () => {
    const generateCalendarDays = () => {
      const year = selectedDate.getFullYear();
      const month = selectedDate.getMonth();
      const firstDay = new Date(year, month, 1).getDay();
      const daysInMonth = new Date(year, month + 1, 0).getDate();
      
      const days = [];
      for(let i=0; i<firstDay; i++) days.push(null);
      for(let i=1; i<=daysInMonth; i++) days.push(new Date(year, month, i));
      return days;
    };

    const days = generateCalendarDays();
    const dateStr = selectedDate.toISOString().split('T')[0];
    const todaysAppts = appointments[dateStr] || [];

    return (
      <div className="tab-container flex-col h-full">
        <div className="calendar-header">
          {selectedDate.toLocaleString('default', { month: 'long', year: 'numeric' })}
        </div>
        <div className="calendar-grid">
          {['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].map(d => <div key={d} className="cal-day-name">{d}</div>)}
          {days.map((date, i) => {
            if (!date) return <div key={`empty-${i}`} className="cal-cell empty"></div>;
            const isSelected = date.getDate() === selectedDate.getDate();
            const dStr = date.toISOString().split('T')[0];
            const hasAppt = !!appointments[dStr] && appointments[dStr].length > 0;
            return (
              <div 
                key={i} 
                className={`cal-cell ${isSelected ? 'selected' : ''} ${hasAppt ? 'has-event' : ''}`}
                onClick={() => setSelectedDate(date)}
              >
                {date.getDate()}
              </div>
            );
          })}
        </div>

        <div className="appointments-section mt-6">
          <h4 className="mb-2 text-sm font-semibold text-muted">Appointments for {selectedDate.toLocaleDateString()}</h4>
          
          <div className="flex-col" style={{gap: '8px', marginBottom: '16px'}}>
            <div className="flex" style={{gap: '8px'}}>
              <input 
                type="time" 
                value={newApptTime} 
                onChange={e => setNewApptTime(e.target.value)} 
                style={{flex: 0.4, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '10px 12px', borderRadius: '8px', fontSize: '14px', outline: 'none'}}
              />
              <input 
                type="text" 
                placeholder="Event title..." 
                value={newApptTitle} 
                onChange={e => setNewApptTitle(e.target.value)} 
                style={{flex: 1, background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '10px 12px', borderRadius: '8px', fontSize: '14px', outline: 'none'}}
              />
            </div>
            <button className="utility-btn w-full flex-center" style={{marginTop: '4px'}} onClick={() => {
              if(newApptTime && newApptTitle) {
                const newApts = appointments[dateStr] ? [...appointments[dateStr]] : [];
                newApts.push({ time: newApptTime, title: newApptTitle });
                
                // Sort by time
                newApts.sort((a,b) => a.time.localeCompare(b.time));
                
                setAppointments({...appointments, [dateStr]: newApts});
                setNewApptTitle(""); setNewApptTime("");
              }
            }}>
              <Plus size={16} /> Set Appointment
            </button>
          </div>

          {todaysAppts.length === 0 ? (
            <div className="text-muted text-sm italic">No appointments for this day.</div>
          ) : (
            <div className="list-widget">
              {todaysAppts.map((apt, i) => (
                <div key={i} className="list-item flex justify-between items-center">
                  <div><span>{apt.time}</span> {apt.title}</div>
                  <button className="del-btn" onClick={() => {
                    const newApts = todaysAppts.filter((_, idx) => idx !== i);
                    setAppointments({...appointments, [dateStr]: newApts});
                  }}><Trash2 size={16}/></button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderReminders = () => {
    return (
      <div className="tab-container">
        <div className="add-item-row mb-4">
          <input 
            type="text" 
            placeholder="New reminder..." 
            value={newReminder}
            onChange={e => setNewReminder(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && newReminder.trim()) {
                setReminders([...reminders, { id: Date.now(), text: newReminder, done: false }]);
                setNewReminder("");
              }
            }}
          />
          <button className="icon-btn" onClick={() => {
            if (newReminder.trim()) {
              setReminders([...reminders, { id: Date.now(), text: newReminder, done: false }]);
              setNewReminder("");
            }
          }}><Plus size={18} /></button>
        </div>
        <div className="list-widget">
          {reminders.map(rem => (
            <div key={rem.id} className="checkbox-item flex justify-between">
              <label className="flex items-center gap-3 cursor-pointer" style={{opacity: rem.done ? 0.5 : 1}}>
                <input 
                  type="checkbox" 
                  checked={rem.done} 
                  onChange={() => setReminders(reminders.map(r => r.id === rem.id ? {...r, done: !r.done} : r))}
                />
                <span style={{textDecoration: rem.done ? 'line-through' : 'none'}}>{rem.text}</span>
              </label>
              <button className="del-btn" onClick={() => setReminders(reminders.filter(r => r.id !== rem.id))}><Trash2 size={16}/></button>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderAlarms = () => {
    return (
      <div className="tab-container">
        <div className="add-item-row mb-4">
          <input 
            type="time" 
            value={newAlarmTime}
            onChange={e => setNewAlarmTime(e.target.value)}
          />
          <button className="icon-btn" onClick={() => {
            if (newAlarmTime) {
              setAlarms([...alarms, { id: Date.now(), time: newAlarmTime, active: true }]);
              setNewAlarmTime("");
            }
          }}><Plus size={18} /></button>
        </div>
        <div className="list-widget">
          {alarms.sort((a,b) => a.time.localeCompare(b.time)).map(alarm => (
            <div key={alarm.id} className="alarm-item flex justify-between">
              <div className="flex items-center gap-3">
                <span className={`text-xl font-bold ${alarm.active ? 'text-white' : 'text-muted'}`}>{alarm.time}</span>
                <button 
                  className={`toggle-btn ${alarm.active ? 'on' : 'off'}`}
                  onClick={() => setAlarms(alarms.map(a => a.id === alarm.id ? {...a, active: !a.active} : a))}
                >
                  {alarm.active ? "ON" : "OFF"}
                </button>
              </div>
              <button className="del-btn" onClick={() => setAlarms(alarms.filter(a => a.id !== alarm.id))}><Trash2 size={16}/></button>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderSOS = () => (
    <div className="emergency-tab">
      <div className="text-left mb-6">
        <label className="text-sm font-semibold text-muted block mb-2">Guardian Phone Number</label>
        <input 
          type="tel" 
          placeholder="+1 (555) 000-0000"
          value={guardianNumber}
          onChange={e => setGuardianNumber(e.target.value)}
          style={{
            width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)',
            padding: '12px', borderRadius: '8px', color: 'white', fontSize: '15px', outline: 'none'
          }}
        />
      </div>

      <button 
        className={`sos-btn ${sosActive ? 'active' : ''}`}
        onClick={() => {
          if (!guardianNumber) {
            alert("Please enter a Guardian Phone Number first!");
            return;
          }
          if (!sosActive) {
            alert(`🚨 SOS ALERT DEPLOYED!\n\nCalling Guardian at ${guardianNumber}...\nSending emergency text message to ${guardianNumber}...`);
          }
          setSosActive(!sosActive);
        }}
      >
        <AlertTriangle size={32} />
        {sosActive ? "EMERGENCY ALERT SENT" : "ACTIVATE SOS"}
      </button>
      <p className="help-text">Instantly calls and texts the registered guardian.</p>
      
      <div className="location-toggle mt-4">
        <div className="text-left">
          <div className="text-sm font-semibold text-white">Share Live GPS Location</div>
          <div className="text-xs text-muted">Sends tracking link to guardian</div>
        </div>
        <button 
          className={`toggle-btn ${locationShared ? 'on' : 'off'}`}
          onClick={() => {
            if (!guardianNumber) {
              alert("Please enter a Guardian Phone Number first!");
              return;
            }
            if (!locationShared) {
              alert(`📍 LOCATION SHARING ACTIVE!\n\nLive GPS tracking link sent via SMS to ${guardianNumber}`);
            }
            setLocationShared(!locationShared);
          }}
        >
          {locationShared ? "SHARING" : "OFF"}
        </button>
      </div>
    </div>
  );

  return (
    <div className="app-container">
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2 onClick={() => navigate('/')} style={{cursor: 'pointer'}}>☀️ Lumi</h2>
          <button className="control-btn-small" onClick={() => setSidebarOpen(false)}><X size={20} /></button>
        </div>

        <div className="sidebar-tabs">
          <button className={activeTab === 'notepad' ? 'active' : ''} onClick={() => setActiveTab('notepad')}><Edit3 size={16}/> Notes</button>
          <button className={activeTab === 'calendar' ? 'active' : ''} onClick={() => setActiveTab('calendar')}><CalendarIcon size={16}/> Calendar</button>
          <button className={activeTab === 'reminders' ? 'active' : ''} onClick={() => setActiveTab('reminders')}><Bell size={16}/> Reminders</button>
          <button className={activeTab === 'alarm' ? 'active' : ''} onClick={() => setActiveTab('alarm')}><Clock size={16}/> Alarms</button>
          <button className={`danger-tab ${activeTab === 'emergency' ? 'active' : ''}`} onClick={() => setActiveTab('emergency')}><AlertTriangle size={16}/> SOS</button>
        </div>

        <div className="sidebar-content">
          {activeTab === 'notepad' && renderNotepad()}
          {activeTab === 'calendar' && renderCalendar()}
          {activeTab === 'reminders' && renderReminders()}
          {activeTab === 'alarm' && renderAlarms()}
          {activeTab === 'emergency' && renderSOS()}
        </div>
      </div>

      <div className={`main-area ${sidebarOpen ? 'sidebar-open' : ''}`}>
        {!sidebarOpen && (
          <button className="toggle-sidebar-btn absolute-top-left" onClick={() => setSidebarOpen(true)}>
            <Menu size={24} />
          </button>
        )}

        <div className="top-right-controls">
          <button className="theme-toggle-btn" onClick={() => setIsDarkMode(!isDarkMode)}>
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          
          <div className="avatar-dropdown">
            <button className="dropdown-toggle" onClick={() => setShowAvatarMenu(!showAvatarMenu)}>
              {selectedAvatar.name} <ChevronDown size={16} />
            </button>
            {showAvatarMenu && (
              <div className="dropdown-menu">
                {AVATARS.map(avatar => (
                  <div key={avatar.id} className="dropdown-item" onClick={() => { setSelectedAvatar(avatar); setShowAvatarMenu(false); }}>
                    <img src={avatar.img} alt={avatar.name} />
                    {avatar.name}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="interaction-container">
          <div className={`video-section ${!callActive ? 'minimized' : ''}`}>
            <img src={selectedAvatar.img} alt="Avatar" className="avatar-video-img" />
            
            {callActive && (
              <div className="controls-overlay">
                <button className={`call-btn ${micActive ? 'mic-on' : ''}`} onClick={() => setMicActive(!micActive)}>
                  {micActive ? <Mic size={24} color="#000" /> : <MicOff size={24} />}
                </button>
                <button className="call-btn danger" onClick={() => setCallActive(false)}>
                  <PhoneOff size={24} />
                </button>
              </div>
            )}
          </div>

          <div className={`chat-section ${!callActive ? 'expanded' : ''}`}>
            {!callActive && (
              <div className="chat-header">
                <h3>Text Chat</h3>
                <button className="join-call-btn" onClick={() => setCallActive(true)}>
                  <Phone size={16} /> Join Voice Call
                </button>
              </div>
            )}
            
            <div className="chat-messages">
              {messages.length === 0 && <div className="empty-chat">Start typing to chat...</div>}
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>{msg.text}</div>
              ))}
            </div>
            
            <div className="chat-input-area">
              <input 
                type="text" 
                placeholder="Message Lumi..." 
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
              />
              <button onClick={handleSendMessage}><Send size={18} /></button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
