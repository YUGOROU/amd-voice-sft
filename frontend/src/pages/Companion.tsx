import { useState, useEffect, useRef } from 'react';
import { 
  Menu, X, Mic, MicOff, PhoneOff, Phone, Calendar as CalendarIcon,
  Bell, Edit3, AlertTriangle, Clock, Send, ChevronDown, Plus, Trash2, ChevronLeft, Sun, Moon,
  Loader2, Wifi, WifiOff
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Client } from "@gradio/client";

const AVATARS = [
  { id: 'lumi', name: 'Lumi (Your Guide)', img: '/avatars/lumi.png', pitch: 1.2, rate: 1.1, langMatches: ['en-US', 'en-GB'], female: true },
  { id: 'dorothy', name: 'Dorothy (Elderly Black Woman)', img: '/avatars/dorothy.png', pitch: 0.7, rate: 0.8, langMatches: ['en-US', 'en-GB'], female: true },
  { id: 'harold', name: 'Harold (Elderly White Man)', img: '/avatars/harold.png', pitch: 0.6, rate: 0.85, langMatches: ['en-GB', 'en-US'], female: false },
  { id: 'marcus', name: 'Marcus (Young Black Man)', img: '/avatars/marcus.png', pitch: 0.9, rate: 1.0, langMatches: ['en-US'], female: false },
  { id: 'priya', name: 'Priya (Young South Asian Woman)', img: '/avatars/priya.png', pitch: 1.2, rate: 1.05, langMatches: ['en-IN', 'en-GB'], female: true },
  { id: 'carlos', name: 'Carlos (Middle-aged Latino Man)', img: '/avatars/carlos.png', pitch: 0.85, rate: 1.0, langMatches: ['es-US', 'es-ES', 'en-US'], female: false }
];

export default function Companion() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState('notepad');
  const [callActive, setCallActive] = useState(true);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<{role: string, text: string}[]>([]);
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0]);
  const [showAvatarMenu, setShowAvatarMenu] = useState(false);
  const [summaries, setSummaries] = useState<any[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isAiResponding, setIsAiResponding] = useState(false);
  const recognitionRef = useRef<any>(null);

  // Theme State
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('lumi_theme') === 'dark');
  
  // Backend State
  const [gradioClient, setGradioClient] = useState<any>(null);
  const [backendStatus, setBackendStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    async function initClient() {
      try {
        const client = await Client.connect(window.location.origin + "/gradio");
        setGradioClient(client);
        setBackendStatus('connected');
      } catch {
        setBackendStatus('error');
        retryTimer = setTimeout(initClient, 5000);
      }
    }

    initClient();
    return () => { if (retryTimer) clearTimeout(retryTimer); };
  }, []);
  
  const fetchSummaries = () => {
    fetch('/api/summaries')
      .then(res => res.json())
      .then(data => setSummaries(data))
      .catch(err => console.error("Error fetching summaries:", err));
  };

  useEffect(() => {
    if (activeTab === 'summary') {
      fetchSummaries();
    }
  }, [activeTab]);

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
      localStorage.setItem('lumi_theme', 'dark');
    } else {
      document.body.classList.remove('dark-mode');
      localStorage.setItem('lumi_theme', 'light');
    }
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

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

  // SOS State (Legacy/Future)
  // const [guardianNumber, setGuardianNumber] = useState("");
  // const [locationShared, setLocationShared] = useState(false);
  // const [sosActive, setSosActive] = useState(false);

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

  const speakMessage = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const availableVoices = window.speechSynthesis.getVoices();
    let targetVoice = null;
    for (const lang of selectedAvatar.langMatches) {
      const match = availableVoices.find(v => v.lang.includes(lang) && 
        (selectedAvatar.female ? (v.name.includes('Female') || v.name.includes('Samantha') || v.name.includes('Veena')) 
                               : (v.name.includes('Male') || v.name.includes('Daniel') || v.name.includes('Alex'))));
      if (match) {
        targetVoice = match;
        break;
      }
    }
    
    if (!targetVoice) {
      for (const lang of selectedAvatar.langMatches) {
        const match = availableVoices.find(v => v.lang.includes(lang));
        if (match) { targetVoice = match; break; }
      }
    }

    if (targetVoice) utterance.voice = targetVoice;
    utterance.pitch = selectedAvatar.pitch;
    utterance.rate = selectedAvatar.rate;
    
    window.speechSynthesis.speak(utterance);
  };

  const handleSendMessage = async (overrideText?: string) => {
    const userMsg = overrideText || chatInput;
    if(!userMsg.trim()) return;
    
    // Stop recognition immediately if it was running
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch(e) {}
    }

    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setChatInput('');

    if (gradioClient && backendStatus === 'connected') {
      try {
        setIsAiResponding(true);
        const historyForBackend = messages.map(m => ({
          role: m.role === 'ai' ? 'assistant' : 'user',
          content: m.text
        }));

        const result = await gradioClient.predict("lumi_api", [
          userMsg,
          historyForBackend,
          selectedAvatar.id,
          currentSessionId
        ]);

        const [aiResponse, audioData, _avatarTag, finalId, action] = result.data;
        setCurrentSessionId(finalId);
        
        if (action) {
          executeAction(action);
        }
        
        fetchSummaries();
        setMessages(prev => [...prev, { role: 'ai', text: aiResponse }]);

        if (callActive) {
          if (audioData && audioData.startsWith("data:audio")) {
            if (audioRef.current) {
              audioRef.current.src = audioData;
              audioRef.current.onended = () => {
                setIsAiResponding(false);
                if (isListening && recognitionRef.current) {
                  try { recognitionRef.current.start(); } catch(e) {}
                }
              };
              audioRef.current.play().catch(e => {
                console.error("Audio playback error:", e);
                setIsAiResponding(false);
              });
            }
          } else {
            // Fallback to browser TTS if no audio from backend
            speakMessage(aiResponse);
            // Browser TTS doesn't have an easy onended in this wrapper, 
            // so we'll just set it back after a delay or just skip for now
            setIsAiResponding(false);
          }
        } else {
          setIsAiResponding(false);
        }

      } catch (err) {
        console.error("Backend Error:", err);
        const errorMsg = "I'm having a little trouble connecting. Let me try again in a moment.";
        setMessages(prev => [...prev, { role: 'ai', text: errorMsg }]);
        speakMessage(errorMsg);
        setIsAiResponding(false);
      }
    } else {
      setTimeout(() => {
        const aiResponse = "I'm still connecting to the server. Please wait a moment and try again.";
        setMessages(prev => [...prev, { role: 'ai', text: aiResponse }]);
        speakMessage(aiResponse);
        setIsAiResponding(false);
      }, 1000);
    }
  };

  const loadSession = (session: any) => {
    const history = session.history || [];
    // Convert backend history format (role/content) to frontend format (role/text)
    const frontendHistory = history.map((m: any) => ({
      role: m.role === 'assistant' ? 'ai' : 'user',
      text: m.content.replace('🎤 ', '') // Remove the mic emoji if present
    }));
    setMessages(frontendHistory);
    setCurrentSessionId(session.id);
    setCallActive(false); // Switch to text mode to show the history clearly
    setSidebarOpen(false); // Close sidebar to focus on chat
  };

  const executeAction = (action: any) => {
    if (!action) return;
    const { type, payload } = action;
    console.log("Executing Action:", type, payload);
    
    switch (type) {
      case 'ADD_NOTE':
        if (payload[0]) {
          setNotes(prev => [{ id: Date.now(), date: new Date().toLocaleDateString(), content: payload[0] }, ...prev]);
        }
        break;
      case 'ADD_CALENDAR':
        if (payload[0] && payload[1]) {
          const date = payload[0]; // YYYY-MM-DD
          const title = payload[1];
          setAppointments(prev => {
            const current = prev[date] || [];
            return { ...prev, [date]: [...current, { time: '09:00 AM', title }] };
          });
        }
        break;
      case 'ADD_REMINDER':
        if (payload[0]) {
          setReminders(prev => [{ id: Date.now(), text: payload[0], done: false }, ...prev]);
        }
        break;
      case 'ADD_ALARM':
        if (payload[0]) {
          setAlarms(prev => [...prev, { id: Date.now(), time: payload[0], active: true }]);
        }
        break;
      default:
        console.warn("Unknown action type:", type);
    }
  };

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event: any) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript) {
          console.log("Transcribed speech:", transcript);
          handleSendMessage(transcript);
        }
      };

      recognition.onend = () => {
        // Only restart if listening is ON and AI is NOT talking
        if (isListening && !isAiResponding) {
          try { recognition.start(); } catch(e) { /* already started */ }
        }
      };

      recognitionRef.current = recognition;
    }
  }, [isListening, isAiResponding]);

  useEffect(() => {
    if (isListening && !isAiResponding) {
      try { recognitionRef.current?.start(); } catch(e) { /* ignore */ }
    } else {
      recognitionRef.current?.stop();
    }
  }, [isListening, isAiResponding]);

  const starterChips = [
    "I can't find my glasses.",
    "Tell me about my family.",
    "What should I do today?",
    "I'm feeling a bit lonely."
  ];

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
    <div className="tab-container flex-col" style={{ gap: '16px' }}>
      <div className="emergency-tab">
        <div className="text-center p-4">
          <h2 className="text-xl font-bold mb-2">Emergency SOS</h2>
          <p className="text-muted italic">This feature is currently under development to ensure 100% reliability for emergency situations.</p>
        </div>
        
        <button 
          className="sos-btn"
          style={{ opacity: 0.5, cursor: 'not-allowed' }}
          disabled
        >
          <AlertTriangle size={48} />
          COMING SOON
        </button>
      </div>
    </div>
  );

  const renderSummaries = () => (
    <div className="tab-container flex-col" style={{ gap: '16px' }}>
      <div className="mb-2">
        <h3 className="text-white font-bold">Session Summaries</h3>
        <p className="text-xs text-muted">Lumi's cognitive logs and history.</p>
      </div>
      
      <div className="summary-list flex-col" style={{ gap: '12px', overflowY: 'auto', maxHeight: '500px', paddingRight: '4px' }}>
        {summaries.length === 0 && <p className="text-xs text-muted italic">No past sessions recorded yet.</p>}
        {summaries.map((s, i) => (
          <div 
            key={i} 
            className="history-card clickable-history" 
            onClick={() => loadSession(s)}
          >
            <div className="history-date flex justify-between">
              <span>{new Date(s.timestamp).toLocaleDateString()}</span>
              <span className="mood-tag">{s.emotional_state}</span>
            </div>
            <p className="text-sm mt-2">{s.summary}</p>
            <div className="resume-hint text-xs mt-2 italic">Click to resume conversation...</div>
            {s.facts && s.facts.length > 0 && (
              <div className="mt-2 pt-2 border-top flex flex-wrap gap-1">
                {s.facts.map((f: string, fi: number) => (
                  <span key={fi} className="fact-pill text-xs">{f}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className={`app-container ${isDarkMode ? 'dark-mode' : ''}`}>
      {/* Sidebar Backdrop for Mobile */}
      <div 
        className={`sidebar-backdrop ${sidebarOpen ? 'active' : ''}`} 
        onClick={() => setSidebarOpen(false)}
      ></div>

      <aside className={`sidebar ${sidebarOpen ? '' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="flex items-center gap-3">
            <h2 onClick={() => navigate('/')} style={{cursor: 'pointer'}}>☀️ Lumi</h2>
            <div className={`status-pill ${backendStatus}`} title={`Backend: ${backendStatus}`}>
              {backendStatus === 'connecting' && <Loader2 size={12} className="spin-slow" />}
              {backendStatus === 'connected' && <Wifi size={12} />}
              {backendStatus === 'error' && <WifiOff size={12} />}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="control-btn-small" onClick={toggleDarkMode}>
              {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button className="control-btn-small mobile-only" onClick={() => setSidebarOpen(false)}>
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="sidebar-tabs">
          <button className="new-chat-btn" onClick={() => { setMessages([]); setCurrentSessionId(null); setActiveTab('notepad'); }}>
            <Plus size={16} /> New Chat
          </button>
          <button className={activeTab === 'notepad' ? 'active' : ''} onClick={() => setActiveTab('notepad')}><Edit3 size={16}/> Notes</button>
          <button className={activeTab === 'calendar' ? 'active' : ''} onClick={() => setActiveTab('calendar')}><CalendarIcon size={16}/> Calendar</button>
          <button className={activeTab === 'reminders' ? 'active' : ''} onClick={() => setActiveTab('reminders')}><Bell size={16}/> Reminders</button>
          <button className={activeTab === 'alarm' ? 'active' : ''} onClick={() => setActiveTab('alarm')}><Clock size={16}/> Alarms</button>
          <button className={activeTab === 'summary' ? 'active' : ''} onClick={() => setActiveTab('summary')}><Clock size={16}/> Summary</button>
          <button className={`danger-tab ${activeTab === 'emergency' ? 'active' : ''}`} onClick={() => setActiveTab('emergency')}>
            <AlertTriangle size={16}/> SOS <span className="coming-soon-badge">Soon</span>
          </button>
        </div>

        <div className="sidebar-content">
          {activeTab === 'notepad' && renderNotepad()}
          {activeTab === 'calendar' && renderCalendar()}
          {activeTab === 'reminders' && renderReminders()}
          {activeTab === 'alarm' && renderAlarms()}
          {activeTab === 'summary' && renderSummaries()}
          {activeTab === 'emergency' && renderSOS()}
        </div>
      </aside>

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
            <img 
              src={selectedAvatar.img} 
              alt="Avatar Video Stream" 
              className="avatar-video-img"
              style={{
                objectFit: selectedAvatar.id === 'lumi' ? 'contain' : 'cover',
                padding: selectedAvatar.id === 'lumi' ? '24px' : '0px'
              }}
            />
            
            {callActive && (
              <div className="controls-overlay">
                <button className={`call-btn ${isListening ? 'mic-on' : ''}`} onClick={() => setIsListening(!isListening)}>
                  {isListening ? <Mic size={24} color="#000" /> : <MicOff size={24} />}
                </button>
                <button className="call-btn danger" onClick={() => { setCallActive(false); setIsListening(false); }}>
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
              {messages.length === 0 && (
                <div className="empty-chat-container">
                  <div className="empty-chat">Choose a prompt to begin...</div>
                  <div className="starter-chips">
                    {starterChips.map((chip, i) => (
                      <button key={i} className="chip" onClick={() => handleSendMessage(chip)}>{chip}</button>
                    ))}
                  </div>
                </div>
              )}
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
              <button onClick={() => handleSendMessage()}><Send size={18} /></button>
            </div>
          </div>
        </div>
      </div>
      {/* Hidden Audio for Backend TTS */}
      <audio ref={audioRef} hidden />
    </div>
  );
}
