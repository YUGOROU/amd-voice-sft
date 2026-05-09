import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Settings, Play, ArrowRight, Edit2 } from 'lucide-react';

export default function Landing() {
  const navigate = useNavigate();
  const [user, setUser] = useState({ name: "Guest", avatarUrl: "https://i.pravatar.cc/150?img=8", handle: "" });
  const [isEditingName, setIsEditingName] = useState(false);
  const [tempName, setTempName] = useState("");

  useEffect(() => {
    const saved = localStorage.getItem('lumi_user');
    if (saved) {
      const parsedUser = JSON.parse(saved);
      setUser(parsedUser);
      setTempName(parsedUser.name);
    } else {
      setTempName("Guest");
    }
  }, []);

  const handleSaveName = () => {
    const updatedUser = { ...user, name: tempName };
    setUser(updatedUser);
    localStorage.setItem('lumi_user', JSON.stringify(updatedUser));
    setIsEditingName(false);
  };

  return (
    <div className="landing-container">
      <nav className="landing-nav">
        <div className="nav-logo">☀️ Lumi</div>
        <div className="nav-profile">
          <img src={user.avatarUrl} alt="Profile" className="nav-avatar" />
        </div>
      </nav>

      <main className="landing-main">
        <div className="welcome-section">
          <div className="profile-large">
            <img src={user.avatarUrl} alt="Profile" />
          </div>
          
          <div className="greeting-container">
            <h1>Good day, 
              {isEditingName ? (
                <div className="name-edit-inline">
                  <input 
                    autoFocus
                    value={tempName} 
                    onChange={(e) => setTempName(e.target.value)} 
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                  />
                  <button onClick={handleSaveName}>Save</button>
                </div>
              ) : (
                <span className="user-name" onClick={() => setIsEditingName(true)}>
                  {user.name} <Edit2 size={16} className="edit-icon" />
                </span>
              )}
            </h1>
            <p>I'm ready whenever you'd like to chat.</p>
          </div>

          <div className="action-cards">
            <div className="action-card primary" onClick={() => navigate('/app')}>
              <div className="card-icon"><Play size={24} /></div>
              <div className="card-text">
                <h3>Start Session</h3>
                <p>Begin a voice or text chat</p>
              </div>
              <ArrowRight className="card-arrow" />
            </div>

            <div className="action-card secondary">
              <div className="card-icon"><Settings size={24} /></div>
              <div className="card-text">
                <h3>Settings</h3>
                <p>Configure guardian alerts & preferences</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
