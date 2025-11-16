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
  target_string: string;
  current_word_index: number;
  current_char_target_sequence: string[];
  current_char_next_digit_index: number;
}

interface DetectionData {
  left_angle: number | null;
  right_angle: number | null;
  current_digit: number | null;
  sequence: string[];
  display_result: string | null;
  state: string;
  prompt_code: string | null;
  l_arm_status: string;
  r_arm_status: string;
  timer: number;
  target_person_bbox: number[] | null;
  flag_boxes: number[][];
  mode: string;
  exam_info: ExamInfo;
  cross_count: number;
  word_history: string[];
  challenge_info: ChallengeInfo;
  gesture_angle_threshold: number;
  gesture_cross_threshold: number;
}

const promptMessages: { [key: string]: (data: DetectionData) => string } = {
    'WAITING_FOR_PERSON': () => '偵測中，請確保您在鏡頭前方...',
    'GESTURE_START_PROMPT': () => '請舉起雙手，開始啟動手勢',
    'AWAITING_HIGH_WAVE': (data) => `請向上揮動，手肘可微彎 (${data.cross_count}/${data.gesture_cross_threshold})`,
    'AWAITING_CROSS': (data) => `請交叉手腕 (${data.cross_count}/${data.gesture_cross_threshold})`,
    'AWAITING_LOW_WAVE': (data) => `請伸直雙手，向下揮動 (${data.cross_count}/${data.gesture_cross_threshold})`,
    'GESTURE_COMPLETE': (data) => `手勢完成！請放下雙手以${data.state === 'IDLE' ? '開始' : '重置'}`,
    'SYSTEM_ACTION_TRIGGERED': (data) => `收到指令，系統${data.state === 'IDLE' ? '開始' : '重置'}！`,
    'PRACTICE_WAITING': () => '請將雙手垂直放下，進入準備姿勢',
    'PRACTICE_READY': () => '準備就緒，請開始動作',
    'PRACTICE_DETECTING': (data) => `偵測到指令 ${data.current_digit}，請保持穩定...`,
    'PRACTICE_ARMS_NOT_STRAIGHT': () => '手臂未伸直，此次辨識無效，請重做',
    'PRACTICE_COOLDOWN': () => '成功！請放下雙手，準備下一個數字',
    'DELETED_LAST_DIGIT': () => '已刪除上一個數字',
    'DELETED_LAST_WORD': () => '已刪除上一個文字',
};

const getPromptText = (data: DetectionData | null): string => {
    if (!data || !data.prompt_code) {
        return '-';
    }
    const knownMessages: { [key: string]: string } = {
        '已刪除上一個數字': '已刪除上一個數字',
        '已刪除上一個文字': '已刪除上一個文字',
    };
    if (knownMessages[data.prompt_code]) {
        return knownMessages[data.prompt_code];
    }
    const messageFunc = promptMessages[data.prompt_code];
    if (messageFunc) {
        return messageFunc(data);
    }
    return data.prompt_code;
};


const VideoStream: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [currentMode, setCurrentMode] = useState<string>('practice');
  const [examTargetSequence, setExamTargetSequence] = useState<string>('1234');
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Updated states for Challenge Mode
  const [isChallengeEnabled, setIsChallengeEnabled] = useState<boolean>(false);
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
      console.log('WebSocket Connected');
      setBackendError(null);
      sendMessage('set_mode', { mode: currentMode, target_sequence: examTargetSequence.split('') });
      sendMessage('set_video_source', { source: '0' });
      setIsLoading(false);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);

        if (payload.error) {
          setBackendError(payload.error);
          setIsLoading(false);
          setDetectionData(null);
          const canvas = canvasRef.current;
          if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
          }
          return;
        }
        
        if (payload.status) {
          console.log("Backend Status:", payload.status);
          setBackendError(null);
          setIsLoading(true);
          return;
        }

        setBackendError(null);
        setIsLoading(false);
        const data = payload.data;
        setDetectionData(data);

        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = `data:image/jpeg;base64,${payload.image}`;
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        setBackendError(`Failed to parse message from backend: ${error}`);
        setIsLoading(false);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsLoading(false);
      setBackendError("WebSocket disconnected. Please ensure the backend server is running.");
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setIsLoading(false);
      setBackendError(`WebSocket error: ${error}`);
    };

    return () => {
      ws.close();
    };
  }, [sendMessage, currentMode, examTargetSequence]);


  const handleModeChange = (mode: string) => {
    setIsLoading(true);
    setCurrentMode(mode);
    if (mode !== 'practice') {
      setIsChallengeEnabled(false);
      sendMessage('set_challenge_mode', { enabled: false });
    }
    sendMessage('set_mode', { mode: mode, target_sequence: examTargetSequence.split('') });
  };

  const handleToggleChallengeMode = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setIsChallengeEnabled(isEnabled);
    sendMessage('set_challenge_mode', { enabled: isEnabled, chars: isEnabled ? challengeString : null });
  };

  const handleSetChallengeString = () => {
    if (isChallengeEnabled && challengeString) {
      sendMessage('set_challenge_mode', { enabled: true, chars: challengeString });
    }
  };

  const handleStartExam = () => {
    setIsLoading(true);
    sendMessage('start_exam');
  };

  const handleStopExam = () => {
    setIsLoading(true);
    sendMessage('stop_exam');
  };

  const challengeInfo = detectionData?.challenge_info;
  const nextDigit = challengeInfo?.is_challenge_mode && challengeInfo.current_char_target_sequence.length > 0
    ? challengeInfo.current_char_target_sequence[challengeInfo.current_char_next_digit_index]
    : null;

  const renderTargetString = () => {
    if (!challengeInfo || !challengeInfo.target_string) return null;
    
    return (
      <p>
        目標: {' '}
        {challengeInfo.target_string.split('').map((char, index) => (
          <strong 
            key={index} 
            style={{ 
              color: index === challengeInfo.current_word_index ? '#FFD700' : 'white',
              fontSize: '1.2em',
              margin: '0 2px'
            }}
          >
            {char}
          </strong>
        ))}
        {' ('}
        {challengeInfo.current_char_target_sequence.join(' ')}
        {')'}
      </p>
    );
  };


  const isDisconnected = backendError && (backendError.includes("WebSocket disconnected") || backendError.includes("WebSocket error"));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px', backgroundColor: '#282c34', color: 'white', minHeight: '100vh' }}>
      <h1>旗語姿態偵測系統</h1>
      
      {backendError && !isDisconnected && (
        <div style={{ color: '#FF4C6C', fontWeight: 'bold', marginBottom: '20px', border: '2px solid #FF4C6C', padding: '10px', borderRadius: '5px' }}>
          Backend Error: {backendError}
        </div>
      )}

      {isDisconnected ? (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          color: 'white',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
          textAlign: 'center',
          fontSize: '1.5em',
          padding: '20px'
        }}>
          <h2>連線已中斷</h2>
          <p>{backendError}</p>
          <button 
            onClick={() => window.location.reload()} 
            style={{ 
              marginTop: '20px', 
              padding: '15px 30px', 
              fontSize: '1em', 
              backgroundColor: '#007bff', 
              color: 'white', 
              border: 'none', 
              borderRadius: '5px', 
              cursor: 'pointer' 
            }}
          >
            重新整理頁面
          </button>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '20px', marginBottom: '20px', padding: '20px', backgroundColor: '#333', borderRadius: '8px', width: '100%', maxWidth: '1280px' }}>
            
            <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h3>模式選擇</h3>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button onClick={() => handleModeChange('practice')} style={{ padding: '10px 20px', backgroundColor: currentMode === 'practice' ? '#007bff' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                  練習模式
                </button>
                <button onClick={() => handleModeChange('exam')} style={{ padding: '10px 20px', backgroundColor: currentMode === 'exam' ? '#007bff' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                  考試模式
                </button>
              </div>
            </div>

            {currentMode === 'practice' && (
              <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '15px', padding: '15px', border: '1px solid #555', borderRadius: '8px' }}>
                <h4>練習模式選項</h4>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '1.1em' }}>
                  <input type="checkbox" checked={isChallengeEnabled} onChange={handleToggleChallengeMode} style={{ width: '20px', height: '20px', marginRight: '10px' }} />
                  指定練習
                </label>
                
                {isChallengeEnabled && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', gap: '10px' }}>
                      <input 
                        type="text" 
                        placeholder="輸入想練習的字串"
                        value={challengeString}
                        onChange={(e) => setChallengeString(e.target.value)}
                        style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', color: 'black', flex: 1 }}
                      />
                      <button onClick={handleSetChallengeString} style={{ padding: '8px 15px', backgroundColor: '#17a2b8', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                        設定
                      </button>
                    </div>
                    {renderTargetString()}
                  </div>
                )}
              </div>
            )}

            {currentMode === 'exam' && (
              <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <h3>考試模式選項</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <label htmlFor="examSequence">Sequence:</label>
                  <input 
                    type="text" 
                    id="examSequence" 
                    value={examTargetSequence} 
                    onChange={(e) => setExamTargetSequence(e.target.value)}
                    style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', color: 'black', flex: 1 }}
                  />
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    onClick={handleStartExam} 
                    disabled={detectionData?.exam_info?.running || detectionData?.exam_info?.exam_finished}
                    style={{ padding: '8px 15px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}
                  >
                    Start Exam
                  </button>
                  <button 
                    onClick={handleStopExam} 
                    disabled={!detectionData?.exam_info?.running && !detectionData?.exam_info?.exam_finished}
                    style={{ padding: '8px 15px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}
                  >
                    Stop Exam
                  </button>
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '20px', width: '100%', maxWidth: '1280px', justifyContent: 'center' }}>
            <div style={{ flex: 'none', maxWidth: '960px', border: '1px solid #444', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
              {isLoading && (
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '2em' }}>
                  Loading Video Stream...
                </div>
              )}
              <canvas ref={canvasRef} style={{ width: '100%', height: 'auto', display: 'block' }} />
            </div>

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px', backgroundColor: '#333', padding: '15px', borderRadius: '8px', textAlign: 'left' }}>
              <h2>偵測資訊</h2>
              
              {isChallengeEnabled && (
                <div style={{ padding: '10px', backgroundColor: '#444', borderRadius: '8px', textAlign: 'center' }}>
                  <h4>下個數字提示</h4>
                  {nextDigit !== null ? (
                    <img src={`/digits/${nextDigit}.png`} alt={`Digit ${nextDigit}`} style={{ width: '80%', height: 'auto', maxHeight: '209px' }} />
                  ) : (
                    <p>請先設定練習字串</p>
                  )}
                </div>
              )}

              {detectionData ? (
                <div style={{ fontSize: '0.9em' }}>
                  <p><strong>提示:</strong> <span style={{ color: '#FF4C6C', fontWeight: 'bold' }}>{getPromptText(detectionData)}</span></p>
                  <hr style={{ borderColor: '#555' }} />
                  <div style={{ minHeight: '60px' }}>
                    <h3>歷史記錄</h3>
                    <p style={{ fontSize: '1.2em', letterSpacing: '2px', wordBreak: 'break-all' }}>
                      {detectionData.word_history && detectionData.word_history.length > 0 
                        ? detectionData.word_history.join(' ') 
                        : '暫無歷史記錄'}
                    </p>
                  </div>
                  <hr style={{ borderColor: '#555' }} />
                  {!isChallengeEnabled && currentMode === 'practice' && (
                    <>
                      <p><strong>目前序列:</strong> {detectionData.sequence.join(' ')}</p>
                      <p><strong>辨識結果:</strong> {detectionData.display_result || 'N/A'}</p>
                    </>
                  )}
                  <p><strong>偵測狀態:</strong> {detectionData.state}</p>
                  <p><strong>揮手計數:</strong> <span style={{ color: '#FFD700' }}>{detectionData.cross_count}</span></p>
                </div>
              ) : (
                <p>等待後端資料...</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default VideoStream;
