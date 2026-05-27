import React, { useEffect, useRef, useState, useCallback } from 'react';

// --- Interfaces ---
interface ExamStats {
  total_signals: number;
  correct_signals: number;
}
interface ChallengeInfo {
  is_challenge_mode: boolean;
  challenge_type: string;
  target_string: string;
  current_word_index: number;
  current_char_target_sequence: string[];
  current_char_next_digit_index: number;
  is_error_locked: boolean;
}
interface DetectionData {
  left_angle: number | null;
  right_angle: number | null;
  current_digit: number | string | null;
  sequence: string[];
  display_result: string | null;
  state: string;
  prompt_code: string | null;
  l_arm_status: string;
  r_arm_status: string;
  target_person_bbox: number[] | null;
  flag_boxes: number[][];
  mode: string;
  cross_count: number;
  word_history: string[];
  challenge_info: ChallengeInfo;
  exam_stats?: ExamStats;
  compute_device?: string;
  backend_fps?: number;
}

const promptMessages: { [key: string]: (data: DetectionData) => string } = {
  'WAITING_FOR_PERSON': () => '尋找目標 (旗手)...',
  'GESTURE_START_PROMPT': () => '舉起雙手交叉啟動',
  'PRACTICE_WAITING': () => '雙手放下預備',
  'PRACTICE_READY': () => '準備就緒，開始比劃',
  'PRACTICE_COOLDOWN': () => '成功！請放下雙手',
};

const getPromptText = (data: DetectionData | null): string => {
  if (!data || !data.prompt_code) return '系統準備中...';
  const messageFunc = promptMessages[data.prompt_code];
  return messageFunc ? messageFunc(data) : data.prompt_code;
};

const VideoStream: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioRefs = {
    correct: useRef<HTMLAudioElement>(new Audio('/digits/correct.mp3')),
    success: useRef<HTMLAudioElement>(new Audio('/digits/success.mp3')),
    incorrect: useRef<HTMLAudioElement>(new Audio('/digits/incorrect.mp3')),
    ok: useRef<HTMLAudioElement>(new Audio('/digits/ok.mp3'))
  };

  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  
  const [questions, setQuestions] = useState<{ chinese: string[], navy: string[] }>({ chinese: [], navy: [] });
  const [mapping, setMapping] = useState<{ [key: string]: string }>({});

  const [system, setSystem] = useState<'chinese' | 'navy'>('chinese');
  const [role, setRole] = useState<'sender' | 'receiver'>('sender');
  const [senderMode, setSenderMode] = useState<'free' | 'practice' | 'exam'>('free');
  const [customPracticeString, setCustomPracticeString] = useState<string>('');
  
  const [receiverMode, setReceiverMode] = useState<'learning' | 'exam'>('learning');
  const [receiverTargetString, setReceiverTargetString] = useState('');
  const [receiverCurrentIndex, setReceiverCurrentIndex] = useState(0);
  const [receiverUserAnswer, setReceiverUserAnswer] = useState('');
  const [receiverFeedback, setReceiverFeedback] = useState<{ text: string, type: 'success' | 'error' | 'neutral' } | null>(null);
  const [isReceiverActive, setIsReceiverActive] = useState(false);
  const [receiverOptions, setReceiverOptions] = useState<string[]>([]);
  const [receiverCorrectCount, setReceiverCorrectCount] = useState(0);
  const [receiverTotalAttempts, setReceiverTotalAttempts] = useState(0);
  const [receiverHasErrored, setReceiverHasErrored] = useState(false);

  const [isFlagRequired, setIsFlagRequired] = useState<boolean>(true);
  const [isMirrored, setIsMirrored] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isDictionaryOpen, setIsDictionaryOpen] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [dictionarySearch, setDictionarySearch] = useState('');
  const [resultOverlay, setResultOverlay] = useState<{ title: string, content: string } | null>(null);

  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraIndex, setSelectedCameraIndex] = useState<number>(0);

  const lastStateRef = useRef<string>('');
  const lastErrorLockRef = useRef<boolean>(false);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        navigator.mediaDevices.enumerateDevices().then(devices => {
          setCameras(devices.filter(device => device.kind === 'videoinput'));
        });
        stream.getTracks().forEach(track => track.stop());
      })
      .catch(err => {
        console.error("Camera access denied:", err);
        navigator.mediaDevices.enumerateDevices().then(devices => {
          setCameras(devices.filter(device => device.kind === 'videoinput'));
        });
      });
  }, []);

  const loadData = useCallback(() => {
    // 使用相對路徑，增加相容性與減少硬編碼 port 的風險
    fetch(`/api/questions`).then(res => res.json()).then(data => {
      setQuestions({ chinese: data.chinese || [], navy: data.navy || [] });
    }).catch(err => {
      console.error('Questions load error:', err);
    });
    fetch(`/api/mapping`).then(res => res.json()).then(data => setMapping(data)).catch(console.error);
  }, []);

  useEffect(() => {
    loadData();
    // 增加一個延遲重試，防止後端啟動較慢時抓不到資料
    const timer = setTimeout(loadData, 2000);
    return () => clearTimeout(timer);
  }, [loadData]);

  const sendMessage = useCallback((command: string, payload?: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, payload }));
    }
  }, []);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
    wsRef.current = ws;
    ws.onopen = () => sendMessage('set_mode', { mode: 'practice', system });
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.image) {
        const img = new Image();
        img.onload = () => {
          const ctx = canvasRef.current?.getContext('2d');
          if (ctx) { canvasRef.current!.width = img.width; canvasRef.current!.height = img.height; ctx.drawImage(img, 0, 0); }
        };
        img.src = `data:image/jpeg;base64,${payload.image}`;
      }
      if (payload.data) {
        const data: DetectionData = payload.data;
        setDetectionData(data);
        if (isAudioEnabled) {
          // 狀態變更時的音效處理
          if (data.state !== lastStateRef.current) {
            if (data.state === 'CHALLENGE_COMPLETE_PROMPT') {
              audioRefs.success.current.play().catch(()=>{});
              setResultOverlay({
                title: data.challenge_info?.challenge_type === 'exam' ? '測驗完成' : '練習完成',
                content: data.word_history.join('')
              });
              setTimeout(() => setResultOverlay(null), 2500);
            } else if (data.state === 'COOLDOWN' || data.state === 'CHALLENGE_AWAITING_GESTURE') {
              if (!data.challenge_info?.is_error_locked) {
                audioRefs.correct.current.play().catch(()=>{});
              }
            }
          }
          // 錯誤鎖定狀態變更時的音效處理 (防止重複播放)
          if (data.challenge_info?.is_error_locked && !lastErrorLockRef.current) {
             audioRefs.incorrect.current.play().catch(()=>{});
          } else if (!data.challenge_info?.is_error_locked && lastErrorLockRef.current && data.current_digit === 'cancel') {
             // 解除鎖定時播正確音效
             audioRefs.correct.current.play().catch(()=>{});
          }
        }
        lastStateRef.current = data.state;
        lastErrorLockRef.current = data.challenge_info?.is_error_locked || false;

      }
    };
    return () => ws.close();
  }, [system, sendMessage, isAudioEnabled]);

  const startSenderExam = () => {
    const bank = questions[system];
    if (bank && bank.length > 0) {
      let testStr = "";
      for (let i = 0; i < 5; i++) {
        testStr += bank[Math.floor(Math.random() * bank.length)];
        if (i < 4) testStr += ",";
      }
      setSenderMode('exam');
      sendMessage('set_challenge_mode', { enabled: true, chars: testStr, type: 'exam' });
    } else alert("題庫載入中...");
  };

  const generateOptions = useCallback((correctChar: string) => {
    let chars: string[] = [];
    if (system === 'navy') {
      chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split('');
    } else {
      chars = Object.keys(mapping).filter(k => k.length === 1 && !/^[A-Z0-9#]$/i.test(k));
    }
    
    const correctSeq = mapping[correctChar] || correctChar;
    // 過濾掉所有與正確答案「動作完全一樣」的其他字元 (例如 D 與 4)
    const validChars = chars.filter(c => c === correctChar || (mapping[c] || c) !== correctSeq);

    const options = new Set<string>();
    options.add(correctChar);
    let attempts = 0;
    while(options.size < 4 && attempts < 100) {
      if (validChars.length > 0) options.add(validChars[Math.floor(Math.random() * validChars.length)]);
      attempts++;
    }
    return Array.from(options).sort(() => Math.random() - 0.5);
  }, [mapping, system]);

  const startReceiver = (mode: 'learning' | 'exam') => {
    if (mode === 'learning') {
      const learnStr = system === 'navy' ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" : "0123456789";
      setReceiverTargetString(learnStr);
      setReceiverCurrentIndex(0);
      setReceiverMode(mode);
      setIsReceiverActive(true);
      setReceiverFeedback(null);
    } else {
      const bank = questions[system];
      if (bank && bank.length > 0) {
        const shuffled = [...bank].sort(() => 0.5 - Math.random());
        const selected = shuffled.slice(0, Math.min(5, shuffled.length));
        const testStr = selected.join('');
        setReceiverTargetString(testStr);
        setReceiverCurrentIndex(0);
        setReceiverMode(mode);
        setIsReceiverActive(true);
        setReceiverFeedback(null);
        setReceiverOptions(generateOptions(testStr[0]));
        setReceiverCorrectCount(0);
        setReceiverTotalAttempts(0);
        setReceiverHasErrored(false);
      } else alert("題庫載入中...");
    }
  };

  const handleReceiverOptionClick = (opt: string) => {
    if (receiverFeedback?.type === 'success') return;
    
    if (receiverMode === 'exam') {
       setReceiverTotalAttempts(prev => prev + 1);
    }
    
    const correctChar = receiverTargetString[receiverCurrentIndex];
    if (opt === correctChar) {
      setReceiverFeedback({ text: '正確！', type: 'success' });
      if (isAudioEnabled) audioRefs.ok.current.play().catch(()=>{});
      
      if (receiverMode === 'exam' && !receiverHasErrored) {
          setReceiverCorrectCount(prev => prev + 1);
      }
      
      setTimeout(() => {
        const nextIdx = receiverCurrentIndex + 1;
        if (nextIdx < receiverTargetString.length) {
          setReceiverCurrentIndex(nextIdx);
          setReceiverFeedback(null);
          setReceiverHasErrored(false);
          if (receiverMode === 'exam') setReceiverOptions(generateOptions(receiverTargetString[nextIdx]));
        } else {
          setResultOverlay({
            title: receiverMode === 'exam' ? '測驗完成' : '學習完成',
            content: receiverMode === 'exam' ? '恭喜過關' : '教學結束'
          });
          setTimeout(() => {
            setResultOverlay(null);
            setIsReceiverActive(false);
          }, 2500);
          if (isAudioEnabled) audioRefs.success.current.play().catch(()=>{});
        }
      }, 1000);
    } else {
      setReceiverFeedback({ text: '錯誤！', type: 'error' });
      if (receiverMode === 'exam') setReceiverHasErrored(true);
      if (isAudioEnabled) audioRefs.incorrect.current.play().catch(()=>{});
    }
  };

  useEffect(() => {
    if (role === 'receiver' && isReceiverActive) {
      if (receiverMode === 'learning') {
        const learnStr = system === 'navy' ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" : "0123456789";
        setReceiverTargetString(learnStr);
        setReceiverCurrentIndex(0);
        setReceiverFeedback(null);
      } else {
        const bank = questions[system];
        if (bank && bank.length > 0) {
          const shuffled = [...bank].sort(() => 0.5 - Math.random());
          const selected = shuffled.slice(0, Math.min(5, shuffled.length));
          const testStr = selected.join('');
          
          setReceiverTargetString(testStr);
          setReceiverCurrentIndex(0);
          setReceiverFeedback(null);
          setReceiverOptions(generateOptions(testStr[0]));
          setReceiverCorrectCount(0);
          setReceiverHasErrored(false);
        }
      }
    }
  }, [system, role, isReceiverActive, receiverMode, questions, generateOptions]);

  const getHintImage = () => {
    if (!detectionData) return null;
    const info = detectionData.challenge_info;
    if (info?.challenge_type === 'exam') return null;
    
    // 優先判定系統狀態的指示圖
    if (detectionData.state === 'IDLE' || detectionData.state === 'CHALLENGE_READY_TO_END') return '/digits/start&end.png';
    if (detectionData.state === 'WAITING' || detectionData.state === 'COOLDOWN' || detectionData.state === 'CHALLENGE_AWAITING_GESTURE') return '/digits/stay.png';
    
    // 如果處於偵測狀態，才顯示下一個該打的字元
    const next = info?.current_char_target_sequence[info.current_char_next_digit_index];
    if (next) return `/digits/${encodeURIComponent(next)}.png`;
    
    return null;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', backgroundColor: '#121212', color: '#e0e0e0', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      {resultOverlay && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', backgroundColor: 'rgba(40, 167, 69, 0.95)', padding: '40px 80px', borderRadius: '20px', zIndex: 999, textAlign: 'center', animation: 'popInOut 2.5s forwards' }}>
          <h2 style={{ fontSize: '2em', margin: '0 0 10px 0' }}>{resultOverlay.title}</h2>
          <div style={{ fontSize: '5em', fontWeight: 'bold' }}>{resultOverlay.content}</div>
        </div>
      )}
      <style>{`.btn-hover:hover { filter: brightness(1.2); } .tab-active { background-color: #007bff !important; color: white !important; } .sys-active { background-color: #28a745 !important; color: white !important; } @keyframes popInOut { 0% { opacity: 0; transform: translate(-50%, -50%) scale(0.5); } 15% { opacity: 1; transform: translate(-50%, -50%) scale(1.05); } 100% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); } }`}</style>

      <header style={{ height: '60px', backgroundColor: '#1e1e1e', borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', padding: '0 20px', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={{ margin: 0, fontSize: '1.4em', color: '#fff' }}>🚩 旗語辨識教學系統</h1>
          <div style={{ display: 'flex', backgroundColor: '#2d2d2d', borderRadius: '6px', overflow: 'hidden' }}>
            <button className={`btn-hover ${system === 'chinese' ? 'sys-active' : ''}`} onClick={() => setSystem('chinese')} style={{ padding: '6px 15px', background: 'transparent', border: 'none', color: '#aaa', cursor: 'pointer' }}>童軍 (中文)</button>
            <button className={`btn-hover ${system === 'navy' ? 'sys-active' : ''}`} onClick={() => setSystem('navy')} style={{ padding: '6px 15px', background: 'transparent', border: 'none', color: '#aaa', cursor: 'pointer', borderLeft: '1px solid #444' }}>海軍 (英文)</button>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {cameras.length > 0 && (
            <select
              value={selectedCameraIndex}
              onChange={(e) => {
                const newIdx = parseInt(e.target.value);
                setSelectedCameraIndex(newIdx);
                sendMessage('set_camera', { device_id: newIdx.toString() });
              }}
              style={{ padding: '5px', borderRadius: '5px', backgroundColor: '#333', color: 'white', border: '1px solid #555', cursor: 'pointer' }}
            >
              {cameras.map((cam, idx) => (
                <option key={cam.deviceId || idx} value={idx}>
                  {cam.label || `Camera ${idx + 1}`}
                </option>
              ))}
              <option value="-1">關閉攝影機 (Off)</option>
            </select>
          )}
          <label style={{ fontSize: '0.9em' }}><input type="checkbox" checked={isMirrored} onChange={() => setIsMirrored(!isMirrored)} /> 鏡像</label>
          <label style={{ fontSize: '0.9em' }}><input type="checkbox" checked={isAudioEnabled} onChange={() => setIsAudioEnabled(!isAudioEnabled)} /> 音效</label>
          <button className="btn-hover" onClick={() => setIsDictionaryOpen(true)} style={{ padding: '6px 15px', backgroundColor: '#6f42c1', color: 'white', border: 'none', borderRadius: '5px', fontWeight: 'bold', cursor: 'pointer' }}>旗語字典</button>
          <button className="btn-hover" onClick={() => setIsInfoOpen(true)} style={{ width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'transparent', color: '#aaa', border: '2px solid #aaa', fontFamily: 'serif', fontStyle: 'italic', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2em' }}>i</button>
        </div>
      </header>

      <div style={{ flex: 1, display: 'flex', padding: '15px', gap: '15px', minHeight: 0 }}>
        <aside style={{ width: '320px', display: 'flex', flexDirection: 'column', gap: '15px', flexShrink: 0 }}>
          <div style={{ backgroundColor: '#1e1e1e', borderRadius: '10px', padding: '15px', border: '1px solid #333' }}>
            <h3 style={{ margin: '0 0 10px 0', fontSize: '1.1em', color: '#aaa' }}>選擇身分</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button className={`btn-hover ${role === 'sender' ? 'tab-active' : ''}`} onClick={() => { setRole('sender'); setIsReceiverActive(false); }} style={{ flex: 1, padding: '10px', background: '#2d2d2d', border: '1px solid #444', borderRadius: '6px', color: '#ccc', cursor: 'pointer' }}>揮旗手 (Sender)</button>
              <button className={`btn-hover ${role === 'receiver' ? 'tab-active' : ''}`} onClick={() => { setRole('receiver'); sendMessage('set_challenge_mode', { enabled: false }); }} style={{ flex: 1, padding: '10px', background: '#2d2d2d', border: '1px solid #444', borderRadius: '6px', color: '#ccc', cursor: 'pointer' }}>觀察員 (Receiver)</button>
            </div>
          </div>

          <div style={{ backgroundColor: '#1e1e1e', borderRadius: '10px', padding: '15px', border: '1px solid #333', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '1.1em', color: '#aaa' }}>操作面板</h3>
            {role === 'sender' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <div style={{ background: '#252525', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#007bff', lineHeight: '1.2' }}>1. 自由練習 <span style={{ fontSize: '0.85em', fontWeight: 'normal', display: 'block', opacity: 0.8 }}>(無限制)</span></div>
                  <button className="btn-hover" onClick={() => { setSenderMode('free'); sendMessage('set_challenge_mode', { enabled: false }); }} style={{ width: '100%', padding: '10px', background: senderMode === 'free' && !detectionData?.challenge_info?.is_challenge_mode && detectionData?.state !== 'IDLE' ? '#007bff' : '#444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>開始自由練習</button>
                </div>
                <div style={{ background: '#252525', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#ccc', lineHeight: '1.2' }}>2. 指定練習 <span style={{ fontSize: '0.85em', fontWeight: 'normal', display: 'block', opacity: 0.8 }}>(有提示)</span></div>
                  <input type="text" value={customPracticeString} onChange={e => setCustomPracticeString(system === 'navy' ? e.target.value.toUpperCase() : e.target.value)} placeholder="輸入字串..." style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #555', background: '#111', color: 'white', marginBottom: '8px', boxSizing: 'border-box' }} />
                  <button className="btn-hover" onClick={() => { setSenderMode('practice'); sendMessage('set_challenge_mode', { enabled: true, chars: customPracticeString, type: 'teaching' }); }} style={{ width: '100%', padding: '8px', background: senderMode === 'practice' && detectionData?.challenge_info?.is_challenge_mode ? '#17a2b8' : '#444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>開始指定練習</button>
                </div>
                <div style={{ background: '#252525', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#dc3545', lineHeight: '1.2' }}>3. 隨機測驗 <span style={{ fontSize: '0.85em', fontWeight: 'normal', display: 'block', opacity: 0.8 }}>(無提示)</span></div>
                  <button className="btn-hover" onClick={startSenderExam} style={{ width: '100%', padding: '10px', background: senderMode === 'exam' && detectionData?.challenge_info?.is_challenge_mode ? '#dc3545' : '#444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>開始抽考</button>
                  {senderMode === 'exam' && detectionData?.exam_stats && (
                    <div style={{ marginTop: '10px', padding: '10px', background: '#111', borderRadius: '4px', fontSize: '0.9em' }}>
                      正確率: {((detectionData.exam_stats.correct_signals / (detectionData.exam_stats.total_signals || 1)) * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <div style={{ background: '#252525', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#17a2b8', lineHeight: '1.2' }}>1. 基礎教學 <span style={{ fontSize: '0.85em', fontWeight: 'normal', display: 'block', opacity: 0.8 }}>(循序)</span></div>
                  <button className="btn-hover" onClick={() => startReceiver('learning')} style={{ width: '100%', padding: '10px', background: isReceiverActive && receiverMode === 'learning' ? '#17a2b8' : '#444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>開始教學認字</button>
                </div>
                <div style={{ background: '#252525', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#dc3545', lineHeight: '1.2' }}>2. 隨機測驗 <span style={{ fontSize: '0.85em', fontWeight: 'normal', display: 'block', opacity: 0.8 }}>(選擇題)</span></div>
                  <button className="btn-hover" onClick={() => startReceiver('exam')} style={{ width: '100%', padding: '10px', background: isReceiverActive && receiverMode === 'exam' ? '#dc3545' : '#444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>開始測驗填字</button>
                  {receiverMode === 'exam' && receiverTotalAttempts > 0 && (
                    <div style={{ marginTop: '10px', padding: '10px', background: '#111', borderRadius: '4px', fontSize: '0.9em' }}>
                      正確率: {((receiverCorrectCount / receiverTotalAttempts) * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </aside>

        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '15px', minWidth: 0 }}>
          <div style={{ height: '60px', backgroundColor: '#1e1e1e', borderRadius: '10px', border: '1px solid #333', display: 'flex', alignItems: 'center', padding: '0 20px', position: 'relative', flexShrink: 0 }}>
            <div style={{ fontSize: '1.4em', fontWeight: 'bold', color: '#FF4C6C' }}>
              {(() => {
                if (role === 'sender') {
                  let prefix = '';
                  const isActive = detectionData && (detectionData.challenge_info?.is_challenge_mode || detectionData.state !== 'IDLE');
                  if (isActive) {
                    if (senderMode === 'exam') prefix = '【隨機測驗】 ';
                    else if (senderMode === 'practice') prefix = '【指定練習】 ';
                    else prefix = '【自由練習】 ';
                  } else {
                    prefix = '【系統待機】 ';
                  }
                  return prefix + getPromptText(detectionData);
                } else {
                  if (!isReceiverActive) return '【系統待機】 請選擇左側模式以開始';
                  if (receiverMode === 'learning') return `【基礎教學】 (${receiverCurrentIndex + 1} / ${receiverTargetString.length})`;
                  return `【隨機測驗】 (${receiverCurrentIndex + 1} / ${receiverTargetString.length})`;
                }
              })()}
            </div>
            {detectionData?.challenge_info?.target_string && role === 'sender' && (
              <div style={{ position: 'absolute', right: '15px', top: '50%', transform: 'translateY(-50%)', display: 'flex', alignItems: 'center', gap: '15px', background: '#000', padding: '6px 15px', borderRadius: '8px', border: '1px solid #444', zIndex: 20, boxShadow: '0 4px 8px rgba(0,0,0,0.5)' }}>
                {(() => {
                   const fullStr = detectionData.challenge_info.target_string;
                   if (fullStr.includes(',')) {
                      const words = fullStr.split(',');
                      let charCount = 0;
                      let currentWordIdx = 0;
                      for (let i = 0; i < words.length; i++) {
                         const nextCount = charCount + words[i].length + 1;
                         if (detectionData.challenge_info.current_word_index < nextCount) {
                             currentWordIdx = i;
                             break;
                         }
                         charCount = nextCount;
                      }
                      if (currentWordIdx < words.length) {
                         return (
                           <>
                             <div style={{ color: '#17a2b8', fontWeight: 'bold', fontSize: '1em', whiteSpace: 'nowrap' }}>第 {currentWordIdx + 1}/5 題</div>
                             <div style={{ width: '1px', height: '25px', background: '#444' }}></div>
                           </>
                         );
                      }
                   }
                   return null;
                })()}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.75em', color: '#888' }}>目標文字</span>
                  <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>
                    {(() => {
                      const fullStr = detectionData.challenge_info.target_string;
                      if (!fullStr.includes(',')) {
                        return fullStr.split('').map((c, i) => (
                          <span key={i} style={{ color: i === detectionData.challenge_info.current_word_index ? '#FFD700' : '#fff' }}>{c}</span>
                        ));
                      }

                      const words = fullStr.split(',');
                      let charCount = 0;
                      let currentWordIdx = 0;
                      let activeCharInWord = -1;

                      for (let i = 0; i < words.length; i++) {
                         const nextCount = charCount + words[i].length + 1;
                         if (detectionData.challenge_info.current_word_index < nextCount) {
                             currentWordIdx = i;
                             activeCharInWord = detectionData.challenge_info.current_word_index - charCount;
                             break;
                         }
                         charCount = nextCount;
                      }

                      if (currentWordIdx >= words.length) return <span style={{color: '#fff'}}>測驗完成</span>;

                      return words[currentWordIdx].split('').map((c, i) => (
                        <span key={i} style={{ color: i === activeCharInWord ? '#FFD700' : '#fff' }}>{c}</span>
                      ));
                    })()}
                  </div>
                </div>
                <div style={{ width: '1px', height: '25px', background: '#444' }}></div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.75em', color: '#888' }}>信號序列</span>
                  <div style={{ fontSize: '1.2em', letterSpacing: '5px' }}>                    {detectionData.challenge_info.current_char_target_sequence.map((s, i) => (
                      <strong key={i} style={{ 
                        color: i === detectionData.challenge_info.current_char_next_digit_index ? '#FFD700' : (i < detectionData.challenge_info.current_char_next_digit_index ? '#333' : '#666'),
                        textDecoration: i < detectionData.challenge_info.current_char_next_digit_index ? 'line-through' : 'none'
                      }}>{s}</strong>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div style={{ flex: 1, backgroundColor: '#000', borderRadius: '10px', border: '1px solid #333', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
            {role === 'sender' ? (
              <>
                <canvas ref={canvasRef} style={{ width: '100%', height: '100%', objectFit: 'contain', transform: isMirrored ? 'scaleX(-1)' : 'none' }} />
                {detectionData && (
                  <div style={{ position: 'absolute', top: '20px', left: '20px', backgroundColor: 'rgba(0,0,0,0.7)', padding: '15px', borderRadius: '10px', color: 'white', fontSize: '1.1em', border: '1px solid #555', zIndex: 10 }}>
                    <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '8px' }}>關節角度偵測</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'auto 60px auto', gap: '8px 12px', alignItems: 'center' }}>
                      <span style={{ whiteSpace: 'nowrap' }}>左手:</span> 
                      <strong style={{ color: '#FFD700', textAlign: 'right' }}>
                        {detectionData.left_angle !== null ? `${Math.round(detectionData.left_angle)}°` : '---'}
                      </strong> 
                      <span style={{ fontSize: '0.85em', whiteSpace: 'nowrap' }}>({detectionData.l_arm_status})</span>
                      
                      <span style={{ whiteSpace: 'nowrap' }}>右手:</span> 
                      <strong style={{ color: '#FFD700', textAlign: 'right' }}>
                        {detectionData.right_angle !== null ? `${Math.round(detectionData.right_angle)}°` : '---'}
                      </strong> 
                      <span style={{ fontSize: '0.85em', whiteSpace: 'nowrap' }}>({detectionData.r_arm_status})</span>
                    </div>
                  </div>
                )}
                <div style={{ position: 'absolute', top: '20px', right: '20px', backgroundColor: 'rgba(0,0,0,0.7)', padding: '10px 15px', borderRadius: '10px', color: 'white', fontSize: '0.9em', border: '1px solid #555', zIndex: 10, display: 'flex', gap: '15px', alignItems: 'center' }}>
                   <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                     <span style={{ color: '#aaa', fontSize: '0.8em' }}>FPS</span>
                     <strong style={{ color: (detectionData?.backend_fps || 0) >= 24 ? '#28a745' : '#FFD700' }}>
                        {detectionData?.backend_fps || 0}
                     </strong>
                   </div>
                   <div style={{ width: '1px', height: '25px', backgroundColor: '#555' }}></div>
                   <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                     <span style={{ color: '#aaa', fontSize: '0.8em' }}>運算單元</span>
                     <strong style={{ color: detectionData?.compute_device === 'CUDA' ? '#28a745' : '#dc3545' }}>
                       {detectionData?.compute_device || '---'}
                     </strong>
                   </div>
                </div>
                {getHintImage() && (
                  <div style={{ position: 'absolute', bottom: '25px', right: '25px', width: '220px', height: '220px', backgroundColor: '#666', borderRadius: '15px', padding: '15px', border: '4px solid #007bff', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
                    <img src={getHintImage()!} alt="Hint" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', transform: isMirrored ? 'scaleX(-1)' : 'none' }} />
                  </div>
                )}
              </>
            ) : (
              isReceiverActive && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', alignItems: 'center', width: '100%' }}>
                  {receiverFeedback && (
                    <div style={{ position: 'absolute', top: '20px', fontSize: '2em', fontWeight: 'bold', color: receiverFeedback.type === 'success' ? '#28a745' : '#dc3545', background: '#fff', padding: '10px 30px', borderRadius: '15px', zIndex: 10 }}>
                      {receiverFeedback.text}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: '15px', background: '#fff', borderRadius: '20px', padding: '25px', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap', maxWidth: '80%' }}>
                    {(() => {
                      const correctChar = receiverTargetString[receiverCurrentIndex];
                      const seq = mapping[correctChar] || correctChar;
                      return seq.split('').map((c, i) => (
                        <img key={i} src={`/digits/${encodeURIComponent(c)}.png`} style={{ width: '120px', height: '120px', objectFit: 'contain' }} alt={c} />
                      ));
                    })()}
                  </div>
                  
                  {receiverMode === 'learning' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
                      <div style={{ fontSize: '5em', color: '#FFD700', fontWeight: 'bold' }}>
                        {receiverTargetString[receiverCurrentIndex]}
                      </div>
                      <div style={{ display: 'flex', gap: '15px' }}>
                        <button className="btn-hover" 
                          onClick={() => setReceiverCurrentIndex(Math.max(0, receiverCurrentIndex - 1))} 
                          disabled={receiverCurrentIndex === 0}
                          style={{ padding: '10px 20px', fontSize: '1.5em', borderRadius: '10px', background: receiverCurrentIndex === 0 ? '#555' : '#17a2b8', color: 'white', border: 'none', cursor: receiverCurrentIndex === 0 ? 'not-allowed' : 'pointer' }}>
                          ⬅️ 上一個
                        </button>
                        <button className="btn-hover" 
                          onClick={() => setReceiverCurrentIndex(Math.min(receiverTargetString.length - 1, receiverCurrentIndex + 1))} 
                          disabled={receiverCurrentIndex === receiverTargetString.length - 1}
                          style={{ padding: '10px 20px', fontSize: '1.5em', borderRadius: '10px', background: receiverCurrentIndex === receiverTargetString.length - 1 ? '#555' : '#17a2b8', color: 'white', border: 'none', cursor: receiverCurrentIndex === receiverTargetString.length - 1 ? 'not-allowed' : 'pointer' }}>
                          下一個 ➡️
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', gap: '20px', width: '80%', maxWidth: '800px', flexWrap: 'wrap', justifyContent: 'center' }}>
                      {receiverOptions.map((opt, i) => (
                        <button key={i} className="btn-hover" onClick={() => handleReceiverOptionClick(opt)} style={{ padding: '20px 40px', fontSize: '2.5em', fontWeight: 'bold', background: '#007bff', color: 'white', border: 'none', borderRadius: '15px', cursor: 'pointer', flex: '1 1 40%' }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            )}
          </div>

          <div style={{ height: '90px', display: 'flex', gap: '15px' }}>
            <div style={{ flex: 1, background: '#1e1e1e', borderRadius: '10px', padding: '10px', border: '1px solid #333', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <span style={{ fontSize: '0.8em', color: '#666', flexShrink: 0 }}>當前信號 (Sequence)</span>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5em', fontWeight: 'bold', color: '#FFD700', letterSpacing: '5px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{detectionData?.sequence.join(' ') || '---'}</div>
            </div>
            <div style={{ flex: 2, background: '#1e1e1e', borderRadius: '10px', padding: '10px', border: '1px solid #333', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <span style={{ fontSize: '0.8em', color: '#666', flexShrink: 0 }}>識別歷史 (History)</span>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', fontSize: '1.1em', color: '#fff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{detectionData?.word_history.join(' ') || '等待中...'}</div>
            </div>
          </div>
        </main>
      </div>

      {isDictionaryOpen && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.95)', zIndex: 1000, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <div style={{ width: '85%', height: '85%', background: '#222', borderRadius: '20px', padding: '35px', display: 'flex', flexDirection: 'column', border: '1px solid #444' }}>
             <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '25px', alignItems: 'center' }}>
               <h2 style={{ color: '#FFD700', margin: 0 }}>📖 旗語字典</h2>
               <button onClick={() => setIsDictionaryOpen(false)} style={{ background: '#dc3545', color: '#fff', border: 'none', borderRadius: '8px', padding: '8px 20px', cursor: 'pointer', fontWeight: 'bold' }}>關閉</button>
             </div>
             <input autoFocus placeholder="搜尋中文字、英文或序號..." value={dictionarySearch} onChange={e => setDictionarySearch(e.target.value)} style={{ width: '100%', padding: '15px', fontSize: '1.3em', borderRadius: '10px', border: 'none', color: '#000', backgroundColor: '#fff', marginBottom: '25px' }} />
             <div style={{ flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '20px' }}>
                {(() => {
                  const searchUpper = dictionarySearch.toUpperCase();
                  let filtered = Object.entries(mapping).filter(([char, seq]) => {
                    if (!dictionarySearch) return /^[A-Z0-9#]$/i.test(char);
                    // 同時將字元與序列(如果有英文字母)轉大寫比對，達到無視大小寫
                    return char.toUpperCase().includes(searchUpper) || 
                           seq.toUpperCase().includes(searchUpper) || 
                           searchUpper.includes(char.toUpperCase());
                  });
                  
                  if (!dictionarySearch) {
                    filtered.sort((a, b) => {
                      const isDigitA = /^\d$/.test(a[0]);
                      const isDigitB = /^\d$/.test(b[0]);
                      if (isDigitA && !isDigitB) return -1;
                      if (!isDigitA && isDigitB) return 1;
                      return a[0].localeCompare(b[0]);
                    });
                  } else if (searchUpper.length > 1 && isNaN(Number(dictionarySearch))) {
                    filtered.sort((a, b) => {
                      const idxA = searchUpper.indexOf(a[0].toUpperCase());
                      const idxB = searchUpper.indexOf(b[0].toUpperCase());
                      if (idxA !== -1 && idxB !== -1) return idxA - idxB;
                      if (idxA !== -1) return -1;
                      if (idxB !== -1) return 1;
                      return 0;
                    });
                  }
                  
                  return filtered.slice(0, 100).map(([char, seq]) => (
                    <div key={char} style={{ background: '#333', padding: '15px', borderRadius: '12px', textAlign: 'center', border: '1px solid #444' }}>
                       <div style={{ fontSize: '2.5em', color: '#FFD700', fontWeight: 'bold' }}>{char}</div>
                       <div style={{ fontSize: '1em', color: '#aaa', marginBottom: '5px' }}>{seq}</div>
                       <div style={{ display: 'flex', justifyContent: 'center', gap: '2px', flexWrap: 'wrap' }}>
                         {seq.split('').map((digit, idx) => (
                           <img key={idx} src={`/digits/${encodeURIComponent(digit)}.png`} style={{ width: '30px', background: '#666', borderRadius: '4px', padding: '2px' }} alt={digit} />
                         ))}
                       </div>
                    </div>
                  ));
                })()}
             </div>
          </div>
        </div>
      )}
      {isInfoOpen && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.95)', zIndex: 1000, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <div style={{ width: '60%', maxHeight: '80%', background: '#222', borderRadius: '20px', padding: '35px', display: 'flex', flexDirection: 'column', border: '1px solid #444', overflowY: 'auto' }}>
             <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '25px', alignItems: 'center' }}>
               <h2 style={{ color: '#FFD700', margin: 0 }}>ℹ️ 旗語簡介與操作說明</h2>
               <button onClick={() => setIsInfoOpen(false)} style={{ background: '#dc3545', color: '#fff', border: 'none', borderRadius: '8px', padding: '8px 20px', cursor: 'pointer', fontWeight: 'bold' }}>關閉</button>
             </div>
             
             <div style={{ lineHeight: '1.6', fontSize: '1.1em', color: '#ddd', textAlign: 'left' }}>
               <h3 style={{ color: '#17a2b8', borderBottom: '1px solid #333', paddingBottom: '10px' }}>什麼是旗語 (Semaphore)?</h3>
               <p style={{ textIndent: '2em' }}>
                 旗語（Flag semaphore）是一種利用手旗或手臂的幾何位置來傳遞視覺信號的通訊系統。它起源於18世紀末的法國，最初由克勞德·查普（Claude Chappe）發明，利用建立在塔頂的機械臂進行長距離通訊。後來演變為我們熟知的手持旗幟版本，被廣泛應用於航海、軍事與童軍活動中。
               </p>
               <p style={{ textIndent: '2em' }}>
                 旗語的優勢在於它不需要電力，只要在視線範圍內即可進行無聲且快速的通訊。國際海軍使用英文旗語（A-Z），而台灣童軍則發展出了一套基於數字（0-9）組合成中文電碼的獨特系統。
               </p>

               <h3 style={{ color: '#17a2b8', borderBottom: '1px solid #333', paddingBottom: '10px', marginTop: '30px' }}>系統使用說明</h3>
               <p style={{ textIndent: '2em' }}>本系統分為兩個主要角色：</p>
               <ul style={{ paddingLeft: '20px' }}>
                 <li style={{ marginBottom: '10px' }}><strong>揮旗手 (Sender)：</strong> 站在攝影機前進行實際揮旗操作。系統會透過 AI 捕捉你的骨架與關節角度，判斷動作是否正確。<ul>
                   <li><strong>自由練習：</strong> 隨意比劃，系統會告訴你現在打出的是什麼信號。</li>
                   <li><strong>指定練習：</strong> 輸入你想要練習的字串，系統會引導你完成。</li>
                   <li><strong>隨機測驗：</strong> 系統隨機出題，考驗你的記憶與動作準確度。</li>
                 </ul></li>
                 <li style={{ marginBottom: '10px' }}><strong>觀察員 (Receiver)：</strong> 學習如何「看懂」別人打的旗語。<ul>
                   <li><strong>基礎教學：</strong> 像是字卡一樣，一步步學習每個基礎信號的長相。</li>
                   <li><strong>測驗填字：</strong> 系統會播放旗語動作，你必須選出對應的字元。</li>
                 </ul></li>
               </ul>

               <h3 style={{ color: '#17a2b8', borderBottom: '1px solid #333', paddingBottom: '10px', marginTop: '30px' }}>打旗語的小訣竅</h3>
               <ul style={{ paddingLeft: '20px' }}>
                 <li><strong>預備姿勢：</strong> 雙手自然下垂於大腿前方交叉。</li>
                 <li><strong>動作俐落：</strong> 揮旗時手臂應盡量伸直，停頓要明確，讓接收者能看清。</li>
                 <li><strong>角度精準：</strong> 旗語是透過兩手的角度差來辨識的（每個方位相隔約 45 度）。使用本系統時，可參考畫面左上角的關節角度提示來調整你的姿勢。</li>
               </ul>
             </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoStream;
