import React, { useEffect, useRef, useState, useCallback } from 'react';

interface ExamInfo {
  target_sequence: string[];
  current_target_digit: string | null;
  correct_count: number;
  total_targets: number;
  time_elapsed: number;
  running: boolean;
  feedback: string;
  exam_finished: boolean;
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

interface CorrectionData {
    target_signal: string | null;
    target_l_angle: number | null;
    target_r_angle: number | null;
    l_angle_diff: number | null;
    r_angle_diff: number | null;
    l_angle_ok: boolean;
    r_angle_ok: boolean;
    l_arm_straight_ok: boolean;
    r_arm_straight_ok: boolean;
    l_advice: string;
    r_advice: string;
    is_correct: boolean;
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
  exam_info: ExamInfo;
  cross_count: number;
  word_history: string[];
  challenge_info: ChallengeInfo;
  correction_data: CorrectionData;
}

const promptMessages: { [key: string]: (data: DetectionData) => string } = {
    'WAITING_FOR_PERSON': () => '尋找目標 (旗手)...',
    'GESTURE_START_PROMPT': () => '舉起雙手交叉啟動',
    'PRACTICE_WAITING': () => '雙手放下預備',
    'PRACTICE_READY': () => '準備就緒，開始比劃',
    'PRACTICE_COOLDOWN': () => '成功！請放下雙手',
};

const getPromptText = (data: DetectionData | null): string => {
    if (!data || !data.prompt_code) return '-';
    const messageFunc = promptMessages[data.prompt_code];
    return messageFunc ? messageFunc(data) : data.prompt_code;
};

const VideoStream: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [currentMode, setCurrentMode] = useState<string>('practice');
  const [practiceSubMode, setPracticeSubMode] = useState<string>('free'); 
  const [currentSystem, setCurrentSystem] = useState<string>('chinese');
  const [isFlagRequired, setIsFlagRequired] = useState<boolean>(true);
  const [examTargetSequence, setExamTargetSequence] = useState<string>('1234');
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isMirrored, setIsMirrored] = useState(true);
  const [challengeString, setChallengeString] = useState<string>('');

  const sendMessage = useCallback((command: string, payload?: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, payload }));
    }
  }, []);

  useEffect(() => {
    setIsLoading(true); 
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;
    ws.onopen = () => {
      setBackendError(null);
      sendMessage('set_mode', { mode: currentMode, system: currentSystem, target_sequence: examTargetSequence.split('') });
      sendMessage('set_video_source', { source: '0' });
      setIsLoading(false);
    };
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.error) { setBackendError(payload.error); setIsLoading(false); return; }
        if (payload.image) {
          const img = new Image();
          img.onload = () => {
            const canvas = canvasRef.current;
            if (canvas) {
              const ctx = canvas.getContext('2d');
              if (ctx) { canvas.width = img.width; canvas.height = img.height; ctx.drawImage(img, 0, 0); }
            }
            setIsLoading(false);
          };
          img.src = `data:image/jpeg;base64,${payload.image}`;
        }
        if (payload.data) setDetectionData(payload.data);
      } catch (err) { console.error('Error:', err); }
    };
    ws.onclose = () => setBackendError("WebSocket disconnected.");
    return () => ws.close();
  }, [currentMode, currentSystem, examTargetSequence, sendMessage]);

  const handleModeChange = (mode: string) => {
    setIsLoading(true);
    setCurrentMode(mode);
    if (mode !== 'practice') sendMessage('set_challenge_mode', { enabled: false });
    sendMessage('set_mode', { mode: mode, system: currentSystem, target_sequence: examTargetSequence.split('') });
  };

  const handleSubModeChange = (sub: string) => {
    setPracticeSubMode(sub);
    if (sub === 'free') {
      sendMessage('set_challenge_mode', { enabled: false });
    } else {
      sendMessage('set_challenge_mode', { enabled: true, chars: challengeString, type: sub });
    }
  };

  const handleSetChallengeString = () => {
    if (challengeString) sendMessage('set_challenge_mode', { enabled: true, chars: challengeString, type: practiceSubMode });
  };

  const handleSystemChange = (system: string) => {
    setIsLoading(true);
    setCurrentSystem(system);
    setChallengeString('');
    sendMessage('set_challenge_mode', { enabled: false });
    sendMessage('set_mode', { mode: currentMode, system: system, target_sequence: examTargetSequence.split('') });
  };

  const challengeInfo = detectionData?.challenge_info;
  const state = detectionData?.state;

  let hintImageSrc: string | null = null;
  if (detectionData) {
    const promptCode = detectionData.prompt_code;
    const nextDigit = challengeInfo?.is_challenge_mode &&
      challengeInfo.current_char_target_sequence.length > 0 &&
      challengeInfo.current_char_next_digit_index < challengeInfo.current_char_target_sequence.length
      ? challengeInfo.current_char_target_sequence[challengeInfo.current_char_next_digit_index]
      : null;

    if (state === 'IDLE' || detectionData.prompt_code?.includes("尋找目標")) {
      hintImageSrc = '/digits/start&end.png';
    }
    else if (state === 'WAITING' || state === 'COOLDOWN' || state === 'CORRECTED_SUCCESS' || state === 'CHALLENGE_AWAITING_GESTURE') {
      hintImageSrc = '/digits/stay.png';
    }
    else if (state === 'CHALLENGE_READY_TO_END') {
      hintImageSrc = '/digits/start&end.png';
    }
    else if (challengeInfo?.is_error_locked) {
      hintImageSrc = (state === 'READY' ? '/digits/cancell.png' : '/digits/stay.png');
    }
    else if (nextDigit !== null && (state === 'READY' || state === 'DETECTING' || state === 'GRACE_PERIOD')) {
      hintImageSrc = `/digits/${nextDigit}.png`;
    }
  }

  const renderTargetString = () => {
    if (!challengeInfo || !challengeInfo.target_string) return null;
    return (
      <div style={{ marginTop: '10px', fontSize: '1.1em', padding: '10px', backgroundColor: '#444', borderRadius: '5px' }}>
        目標: {' '}
        {challengeInfo.target_string.split('').map((char, index) => (
          <strong key={index} style={{ color: index === challengeInfo.current_word_index ? '#FFD700' : 'white', margin: '0 2px' }}>{char}</strong>
        ))}
        <span style={{ marginLeft: '10px', color: '#aaa' }}>
          (
          {challengeInfo.current_char_target_sequence.map((digit, index) => (
            <span key={index} style={{ 
              color: index === challengeInfo.current_char_next_digit_index ? '#FFD700' : 'inherit',
              margin: '0 3px',
              fontWeight: index === challengeInfo.current_char_next_digit_index ? 'bold' : 'normal'
            }}>
              {digit}
            </span>
          ))}
          )
        </span>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px', backgroundColor: '#282c34', color: 'white', minHeight: '100vh', width: '100%' }}>
      <h1>智慧旗語教練系統</h1>
      
      {backendError && (
        <div style={{ color: '#FF4C6C', fontWeight: 'bold', marginBottom: '20px', border: '2px solid #FF4C6C', padding: '10px', borderRadius: '5px' }}>
          Backend Error: {backendError} <button onClick={() => window.location.reload()}>重新整理</button>
        </div>
      )}

      {/* Control Panel */}
      <div style={{ width: '100%', maxWidth: '1600px', padding: '15px', backgroundColor: '#333', borderRadius: '8px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 300px' }}>
            <h3>主要模式</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button onClick={() => handleModeChange('practice')} style={{ padding: '10px', backgroundColor: currentMode === 'practice' ? '#007bff' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', flex: 1, fontWeight: 'bold' }}>練習模式</button>
              <button onClick={() => handleModeChange('exam')} style={{ padding: '10px', backgroundColor: currentMode === 'exam' ? '#dc3545' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', flex: 1, fontWeight: 'bold' }}>考試模式</button>
            </div>
            <h4 style={{ margin: '15px 0 10px 0', color: '#ccc' }}>系統切換</h4>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button onClick={() => handleSystemChange('chinese')} style={{ padding: '8px', backgroundColor: currentSystem === 'chinese' ? '#17a2b8' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', flex: 1 }}>童軍旗語 (中文)</button>
              <button onClick={() => handleSystemChange('navy')} style={{ padding: '8px', backgroundColor: currentSystem === 'navy' ? '#28a745' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', flex: 1 }}>國際旗語 (英文)</button>
            </div>
          </div>
          
          <div style={{ flex: '2 1 500px', borderLeft: '1px solid #555', paddingLeft: '20px' }}>
            <h3>練習子模式</h3>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
              <button onClick={() => handleSubModeChange('free')} style={{ padding: '10px', backgroundColor: practiceSubMode === 'free' ? '#007bff' : '#444', color: 'white', border: 'none', borderRadius: '5px', flex: 1 }}>自由練習</button>
              <button onClick={() => handleSubModeChange('standard')} style={{ padding: '10px', backgroundColor: practiceSubMode === 'standard' ? '#fd7e14' : '#444', color: 'white', border: 'none', borderRadius: '5px', flex: 1 }}>指定練習</button>
              <button onClick={() => handleSubModeChange('teaching')} style={{ padding: '10px', backgroundColor: practiceSubMode === 'teaching' ? '#ffc107' : '#444', color: 'white', border: 'none', borderRadius: '5px', flex: 1, fontWeight: 'bold' }}>教學練習</button>
            </div>
            {practiceSubMode !== 'free' && (
              <div style={{ display: 'flex', gap: '10px' }}>
                <input type="text" placeholder="輸入練習字串..." value={challengeString} onChange={(e) => setChallengeString(currentSystem === 'navy' ? e.target.value.toUpperCase() : e.target.value)} style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ccc', color: 'black', flex: 1 }}/>
                <button onClick={handleSetChallengeString} style={{ padding: '10px 20px', backgroundColor: '#17a2b8', color: 'white', border: 'none', borderRadius: '5px', fontWeight: 'bold' }}>開始練習</button>
              </div>
            )}
          </div>

          <div style={{ flex: '1 1 200px', borderLeft: '1px solid #555', paddingLeft: '20px' }}>
            <h3>全域設定</h3>
            <label style={{ display: 'block', cursor: 'pointer', marginBottom: '10px' }}>
                <input type="checkbox" checked={isFlagRequired} onChange={(e) => { setIsFlagRequired(e.target.checked); sendMessage('set_flag_requirement', { required: e.target.checked }); }} /> 需要旗幟
            </label>
            <label style={{ display: 'block', cursor: 'pointer' }}>
                <input type="checkbox" checked={isMirrored} onChange={() => setIsMirrored(!isMirrored)} /> 鏡像畫面
            </label>
          </div>
        </div>
      </div>

      {/* Main Content (Video & Panel) */}
      <div style={{ display: 'flex', gap: '20px', width: '100%', maxWidth: '1600px', alignItems: 'stretch' }}>
        {/* Video Column */}
        <div style={{ flex: '2 1 0', border: '1px solid #444', borderRadius: '8px', overflow: 'hidden', position: 'relative', backgroundColor: '#000', display: 'flex', alignItems: 'center' }}>
          {isLoading && <div style={{ position: 'absolute', zIndex: 10, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '2em' }}>Loading...</div>}
          <canvas ref={canvasRef} style={{ width: '100%', height: 'auto', display: 'block', transform: isMirrored ? 'scaleX(-1)' : 'none' }} />
        </div>

        {/* Info Column */}
        <div style={{ flex: '1 1 0', backgroundColor: '#333', padding: '20px', borderRadius: '8px', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          {detectionData ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {/* Prompt Section - FIXED HEIGHT to avoid jumping */}
              <div style={{ marginBottom: '20px', minHeight: '150px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
                <p style={{ fontSize: '1.1em', margin: '0 0 5px 0', color: '#aaa' }}><strong>目前提示:</strong></p>
                <div style={{ minHeight: '3em', display: 'flex', alignItems: 'center' }}>
                  <span style={{ color: '#FF4C6C', fontWeight: 'bold', fontSize: '1.7em', lineHeight: '1.2' }}>{getPromptText(detectionData)}</span>
                </div>
                {practiceSubMode !== 'free' && renderTargetString()}
              </div>

              {/* Reference Image Section */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <div style={{ padding: '15px', backgroundColor: '#444', borderRadius: '8px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '220px' }}>
                  <h4 style={{ margin: '0 0 10px 0', color: '#ccc' }}>動作參考</h4>
                  {hintImageSrc ? (
                    <div style={{ width: '100%', height: '180px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <img 
                        src={hintImageSrc} 
                        alt="Hint" 
                        style={{ 
                          maxHeight: '100%', 
                          maxWidth: '100%', 
                          objectFit: 'contain',
                          transform: isMirrored ? 'scaleX(-1)' : 'none',
                          transition: 'transform 0.3s ease' 
                        }} 
                      />
                    </div>
                  ) : <p>無</p>}
                </div>

                <hr style={{ borderColor: '#555', width: '100%' }} />

                {/* Status Data Section */}
                <div style={{ fontSize: '1em' }}>
                  <p style={{ margin: '4px 0' }}><strong>偵測狀態:</strong> {detectionData.state}</p>
                  <p style={{ margin: '4px 0' }}><strong>即時角度:</strong> 左 {detectionData.left_angle?.toFixed(0)}° / 右 {detectionData.right_angle?.toFixed(0)}°</p>
                  <p style={{ margin: '4px 0' }}><strong>手臂狀態:</strong> 左 {detectionData.l_arm_status} / 右 {detectionData.r_arm_status}</p>
                  
                  <p style={{ marginTop: '12px', fontWeight: 'bold', color: '#aaa' }}>歷史記錄:</p>
                  <div style={{ backgroundColor: '#222', padding: '10px', borderRadius: '5px', minHeight: '50px', fontSize: '1.3em', letterSpacing: '3px', wordBreak: 'break-all' }}>
                    {detectionData.word_history?.join(' ') || '暫無記錄'}
                  </div>
                </div>
              </div>
            </div>
          ) : <p>等待後端連線...</p>}
        </div>
      </div>
    </div>
  );
};

export default VideoStream;
