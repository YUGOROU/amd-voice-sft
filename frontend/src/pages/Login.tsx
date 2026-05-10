import { useNavigate } from 'react-router-dom';
import { UserCircle2 } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();

  const handleHFLogin = () => {
    // Mocking the Hugging Face OAuth login process
    const mockUser = {
      name: "Debdeep Banerjee",
      handle: "@debdeep30",
      avatarUrl: "https://i.pravatar.cc/150?img=11" // Mock avatar
    };
    
    // Save to localStorage so we can use it later
    localStorage.setItem('lumi_user', JSON.stringify(mockUser));
    
    // Navigate to landing page
    navigate('/landing');
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h2>Welcome to Lumi</h2>
          <p>Your AI Voice Companion</p>
        </div>
        
        <div className="login-body">
          <button className="hf-login-btn" onClick={handleHFLogin}>
            <span className="hf-logo">🤗</span>
            Sign in with Hugging Face
          </button>
          
          <div className="divider">
            <span>or</span>
          </div>
          
          <button className="guest-login-btn" onClick={() => navigate('/landing')}>
            <UserCircle2 size={18} />
            Continue as Guest
          </button>
        </div>
      </div>
    </div>
  );
}
