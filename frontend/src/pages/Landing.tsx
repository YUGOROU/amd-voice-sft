import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, ArrowRight, Sun, Moon, Sparkles, Info, X, Cpu, MemoryStick, ShieldAlert, Users, Heart, ShieldCheck } from 'lucide-react';

export default function Landing() {
  const navigate = useNavigate();
  const [showAbout, setShowAbout] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('lumi_theme') === 'dark');

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
      localStorage.setItem('lumi_theme', 'dark');
    } else {
      document.body.classList.remove('dark-mode');
      localStorage.setItem('lumi_theme', 'light');
    }
  }, [isDarkMode]);


  return (
    <div className="landing-container">
      <nav className="landing-nav">
        <div className="nav-logo"></div>
        <div className="nav-actions" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button className="theme-toggle-btn" onClick={() => setIsDarkMode(!isDarkMode)}>
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </nav>

      <main className="landing-main-split">
        <div className="left-content">
          <div className="brand-header">
            <div className="lumi-title">Lumi</div>
          </div>
          
          <div className="hero-section">
            <div className="tagline-pill large">
              <Sparkles size={20} style={{ color: 'var(--accent)' }} />
              <span>Compassionate AI Companion for Dementia & Alzheimer's</span>
            </div>
            <p className="hero-subtitle">
              Empathetic, memory-persistent, and specialized care support fine-tuned on AMD hardware.
            </p>
          </div>

          <div className="action-cards-left">
            <div className="action-card primary" onClick={() => navigate('/app')}>
              <div className="card-icon"><Play size={24} /></div>
              <div className="card-text">
                <h3>Start Session</h3>
                <p>Begin a voice or text chat</p>
              </div>
              <ArrowRight className="card-arrow" />
            </div>

            <div className="action-card secondary" onClick={() => setShowAbout(true)}>
              <div className="card-icon"><Info size={24} /></div>
              <div className="card-text">
                <h3>About Lumi</h3>
                <p>Mission, technology & validation</p>
              </div>
            </div>

            <div className="action-card eq-card" style={{ cursor: 'default', border: '1px solid rgba(255, 107, 107, 0.15)' }}>
              <div className="card-icon" style={{ background: 'rgba(255, 107, 107, 0.1)', color: '#ff6b6b' }}>
                <Heart size={24} />
              </div>
              <div className="card-text">
                <h3>91.22 EQ Score</h3>
                <p>Human-level emotional intelligence</p>
              </div>
            </div>

            <div className="action-card safety-card" style={{ cursor: 'default', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div className="card-icon" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981' }}>
                <ShieldCheck size={24} />
              </div>
              <div className="card-text">
                <h3>100% Scam Detection</h3>
                <p>Zero-tolerance safety validation</p>
              </div>
            </div>
          </div>
        </div>

        <div className="right-hero">
          <div className="profile-giant">
            <img src="/avatars/lumi.png" alt="Lumi AI" />
          </div>
        </div>
      </main>

      {showAbout && (
        <div className="about-modal-overlay" onClick={() => setShowAbout(false)}>
          <div className="about-modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowAbout(false)}><X size={24} /></button>
            
            <div className="about-header">
              <h1>Lumi</h1>
              <p className="subtitle">AI Voice Companion for Dementia & Alzheimer's Patients</p>
              <div className="hackathon-badge">
                HuggingFace Spaces | AMD MI300X
              </div>
            </div>

            <div className="about-section">
              <h2>Executive Summary</h2>
              <p>
                Lumi is a fine-tuned AI voice companion designed specifically for elderly patients with dementia and Alzheimer's disease. 
                It combines emotional intelligence, persistent cross-session memory, and a responsive voice-interaction pipeline 
                to provide companionship, cognitive stimulation, and protection against scams.
              </p>
              <p>
                The global dementia population exceeds 55 million people worldwide. Lumi is the first fine-tuned, 
                memory-persistent, voice-native dementia companion built on open models running on AMD hardware.
              </p>
            </div>

            <div className="about-grid">
              <div className="info-box">
                <Cpu size={32} />
                <h3>AMD Native</h3>
                <p>Built on ROCm and vLLM, optimized for AMD Instinct MI300X performance.</p>
              </div>
              <div className="info-box">
                <MemoryStick size={32} />
                <h3>Persistent Memory</h3>
                <p>Uses ChromaDB to track facts across sessions, providing genuine continuity.</p>
              </div>
              <div className="info-box">
                <ShieldAlert size={32} />
                <h3>Scam Protection</h3>
                <p>Dedicated layer to identify and gently deflect fraudulent phone requests.</p>
              </div>
              <div className="info-box">
                <Users size={32} />
                <h3>Family Sync</h3>
                <p>Generates structured summaries for caregivers after every session.</p>
              </div>
            </div>

            <div className="comparison-table-container">
              <h3>How Lumi Compares</h3>
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Existing Apps</th>
                    <th>Lumi</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Domain Fine-Tuning</td>
                    <td>No (Generic LLMs)</td>
                    <td>Yes (QLoRA on Dementia Data)</td>
                  </tr>
                  <tr>
                    <td>Cross-Session Memory</td>
                    <td>No</td>
                    <td>Yes (ChromaDB Facts)</td>
                  </tr>
                  <tr>
                    <td>Voice-Native Architecture</td>
                    <td>Partial</td>
                    <td>Yes (LFM2.5-Audio)</td>
                  </tr>
                  <tr>
                    <td>Animated Avatar</td>
                    <td>Static/Simple</td>
                    <td>5-State Expressive Companion</td>
                  </tr>
                  <tr>
                    <td>Scam Protection</td>
                    <td>No</td>
                    <td>Yes (100% Detection Rate)</td>
                  </tr>
                  <tr>
                    <td>Emotional Intelligence (EQ-Bench)</td>
                    <td>~55 (Generic)</td>
                    <td>91.22 / 100</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="about-section">
              <h2>Safety & Emotional Intelligence</h2>
              <p>
                Lumi has been rigorously validated against industry-standard benchmarks:
              </p>
              <ul style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '8px' }}>
                <li><strong>EQ-Bench v2:</strong> Lumi scored 91.22/100, demonstrating a deep, human-like understanding of emotional nuances.</li>
                <li><strong>Scam Protection:</strong> Lumi achieved a 100% detection rate across all tested fraudulent scenarios.</li>
              </ul>
            </div>

            <div className="about-section">
              <h2>System Architecture</h2>
              <p className="mono-code">
                Mic Input → LFM2.5-Audio → Structured Parser → [avatar_tag] Opening line → TTS (Instant) → Full Response → ChromaDB Write
              </p>
              <p>
                Our "Structured Output Trick" hides latency by firing the opening line immediately while the internal 
                reasoning (<span className="code-tag">think</span> block) is generated. The user hears a response in under 1.5 seconds.
              </p>
            </div>

            <div className="about-footer">
              <p>"This is not a chatbot. This is the companion 55 million people deserve."</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
