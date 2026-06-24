import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  StatusBar,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  Frame,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';

// ── Bridge Return Schema Types ──
interface BoundingBox {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

interface FaceVerifyResult {
  faceDetected: boolean;
  boundingBox: BoundingBox | null;
  isRealFace: boolean;
  livenessScore: number;
  faceEmbedding: number[] | null;
}

import { VisionCameraProxy } from 'react-native-vision-camera';

// Initialize the native frame processor plugin in JSI
const plugin = VisionCameraProxy.initFrameProcessorPlugin('faceVerification');

// ── Map the native frame processor plugin ──
const faceVerification = (frame: Frame): FaceVerifyResult | null => {
  'worklet';
  if (plugin == null) return null;
  // @ts-ignore - native frame processor JSI plugin call
  return plugin.call(frame) as FaceVerifyResult | null;
};

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  // UI States
  const [screen, setScreen] = useState<'WELCOME' | 'SCAN'>('WELCOME');
  const [status, setStatus] = useState<'NO_FACE' | 'SCANNING' | 'VERIFIED' | 'SPOOF'>('NO_FACE');
  const [livenessScore, setLivenessScore] = useState<number>(0);
  const [bbox, setBbox] = useState<BoundingBox | null>(null);
  const [embeddingPreview, setEmbeddingPreview] = useState<string>('Empty');
  const [fps, setFps] = useState<number>(0);

  // Screen Layout Dimensions (App is locked in LANDSCAPE via manifest)
  const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

  // Request permissions on mount
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission]);

  // Frame processor execution telemetry variables
  let lastFrameTime = React.useRef<number>(Date.now());
  let frameTimes = React.useRef<number[]>([]);

  // Telemetry updates to JS Thread
  const updateUIState = Worklets.createRunOnJS((result: FaceVerifyResult | null) => {
    // 1. Compute processing FPS
    const now = Date.now();
    const frameTime = now - lastFrameTime.current;
    lastFrameTime.current = now;
    
    frameTimes.current.push(frameTime);
    if (frameTimes.current.length > 10) {
      frameTimes.current.shift();
    }
    const avgFrameTime = frameTimes.current.reduce((a, b) => a + b, 0) / frameTimes.current.length;
    setFps(Math.round(1000 / avgFrameTime));

    if (!result) {
      return;
    }

    // 2. Map verification states
    if (result.faceDetected) {
      setBbox(result.boundingBox);
      setLivenessScore(result.livenessScore);

      if (result.livenessScore >= 0.5) {
        setStatus('VERIFIED');
      } else {
        setStatus('SPOOF');
      }

      // Preview first 4 embedding values for visual verification
      if (result.faceEmbedding && result.faceEmbedding.length > 0) {
        const previewValues = result.faceEmbedding.slice(0, 4).map(v => v.toFixed(3));
        setEmbeddingPreview(`[${previewValues.join(', ')}, ...]`);
      } else {
        setEmbeddingPreview('Empty');
      }
    } else {
      setStatus('NO_FACE');
      setBbox(null);
      setLivenessScore(0);
      setEmbeddingPreview('Empty');
    }
  });

  // Zero-copy Frame Processor Loop
  const frameProcessor = useFrameProcessor((frame: Frame) => {
    'worklet';
    const result = faceVerification(frame);
    updateUIState(result);
  }, []);

  const renderWelcomeScreen = () => {
    return (
      <View style={[styles.container, styles.welcomeContainer]}>
        <StatusBar hidden />
        <View style={styles.welcomeLeft}>
          <View style={styles.logoContainer}>
            <View style={styles.logoRingOuter}>
              <View style={styles.logoRingInner}>
                <Text style={styles.logoIcon}>🛡️</Text>
              </View>
            </View>
          </View>
          <Text style={styles.welcomeTitle}>VerifyIdentity</Text>
          <Text style={styles.welcomeSub}>On-Device Face Verification & Liveness Pipeline</Text>
        </View>

        <View style={styles.welcomeRight}>
          <View style={styles.healthCard}>
            <Text style={styles.healthTitle}>SYSTEM HEALTH</Text>
            
            <View style={styles.healthRow}>
              <Text style={styles.healthLabel}>Camera Permission:</Text>
              <Text style={[styles.healthVal, hasPermission ? styles.greenText : styles.yellowText]}>
                {hasPermission ? '● Granted' : '● Pending'}
              </Text>
            </View>

            <View style={styles.healthRow}>
              <Text style={styles.healthLabel}>Camera Device:</Text>
              <Text style={[styles.healthVal, device ? styles.greenText : styles.redText]}>
                {device ? '● Ready' : '● Missing'}
              </Text>
            </View>

            <View style={styles.healthRow}>
              <Text style={styles.healthLabel}>Biometric Engine:</Text>
              <Text style={[styles.healthVal, styles.cyanText]}>
                ● TFLite INT8 (Local)
              </Text>
            </View>

            <View style={styles.healthRow}>
              <Text style={styles.healthLabel}>Configuration:</Text>
              <Text style={[styles.healthVal, styles.grayText]}>
                ● Landscape Locked
              </Text>
            </View>
          </View>

          <TouchableOpacity 
            style={styles.scanStartBtn} 
            onPress={() => {
              if (!hasPermission) {
                requestPermission();
              }
              setScreen('SCAN');
            }}
          >
            <Text style={styles.scanStartBtnText}>AUTHENTICATE IDENTITY</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // If we are on the welcome screen, render it directly
  if (screen === 'WELCOME') {
    return renderWelcomeScreen();
  }

  // Permission check only when starting scanning
  if (!hasPermission) {
    return (
      <View style={[styles.container, styles.center]}>
        <StatusBar hidden />
        <Text style={styles.headerText}>Camera Permission Required</Text>
        <Text style={styles.subtext}>VerifyIdentity requires camera access to process biometric verification.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Access</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.button, styles.backBtn, { marginTop: 15 }]} onPress={() => setScreen('WELCOME')}>
          <Text style={styles.buttonText}>Back to Welcome</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={[styles.container, styles.center]}>
        <StatusBar hidden />
        <ActivityIndicator size="large" color="#4a90d9" />
        <Text style={styles.headerText}>Searching for Camera...</Text>
        <TouchableOpacity style={[styles.button, styles.backBtn, { marginTop: 15 }]} onPress={() => setScreen('WELCOME')}>
          <Text style={styles.buttonText}>Back to Welcome</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Bounding Box Layout mappings
  // Normalized bounding boxes are converted to screen layout absolute pixels
  const renderFaceOverlay = () => {
    if (!bbox) return null;

    const overlayWidth = (bbox.xMax - bbox.xMin) * screenWidth;
    const overlayHeight = (bbox.yMax - bbox.yMin) * screenHeight;
    const overlayLeft = bbox.xMin * screenWidth;
    const overlayTop = bbox.yMin * screenHeight;

    const getBorderColor = () => {
      if (status === 'VERIFIED') return '#00ffaa'; // Neon Green
      if (status === 'SPOOF') return '#ff3b30';    // Alert Red
      return '#00e5ff'; // Neon Cyan
    };

    return (
      <View
        style={[
          styles.faceBox,
          {
            width: overlayWidth,
            height: overlayHeight,
            left: overlayLeft,
            top: overlayTop,
            borderColor: getBorderColor(),
          },
        ]}
      >
        {/* Face Bounding Box Corner Indicators */}
        <View style={[styles.corner, styles.topLeft, { borderColor: getBorderColor() }]} />
        <View style={[styles.corner, styles.topRight, { borderColor: getBorderColor() }]} />
        <View style={[styles.corner, styles.bottomLeft, { borderColor: getBorderColor() }]} />
        <View style={[styles.corner, styles.bottomRight, { borderColor: getBorderColor() }]} />
      </View>
    );
  };

  const getStatusBadgeStyle = () => {
    if (status === 'VERIFIED') return styles.verifiedBadge;
    if (status === 'SPOOF') return styles.spoofBadge;
    return styles.noFaceBadge;
  };

  const getStatusText = () => {
    if (status === 'VERIFIED') return 'IDENTITY VERIFIED';
    if (status === 'SPOOF') return 'SPOOF ALERT DETECTED';
    return 'ALIGNING FACE...';
  };

  return (
    <View style={styles.container}>
      <StatusBar hidden />

      {/* Camera Live Stream */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={screen === 'SCAN'}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />

      {/* Face Bounding Box Overlay */}
      {renderFaceOverlay()}

      {/* Glassmorphic Side HUD Panel */}
      <View style={styles.hudPanel}>
        <View style={styles.hudHeader}>
          <TouchableOpacity onPress={() => setScreen('WELCOME')} style={styles.backButton}>
            <Text style={styles.backButtonText}>◀ Exit</Text>
          </TouchableOpacity>
          <View style={styles.liveIndicator}>
            <View style={styles.pulseDot} />
            <Text style={styles.liveText}>LIVE HUD</Text>
          </View>
        </View>

        {/* Verification Status Card */}
        <View style={[styles.hudCard, getStatusBadgeStyle()]}>
          <Text style={styles.hudCardLabel}>VERIFICATION STATUS</Text>
          <Text style={styles.hudCardValue}>{getStatusText()}</Text>
        </View>

        {/* Liveness Analytics Card */}
        <View style={styles.hudCard}>
          <Text style={styles.hudCardLabel}>ANTI-SPOOF LIVENESS</Text>
          <View style={styles.progressContainer}>
            <View style={styles.progressBarBg}>
              <View
                style={[
                  styles.progressBarFill,
                  {
                    width: `${Math.round(livenessScore * 100)}%`,
                    backgroundColor: status === 'VERIFIED' ? '#00ffaa' : status === 'SPOOF' ? '#ff3b30' : '#888',
                  },
                ]}
              />
            </View>
            <Text style={styles.progressValue}>{(livenessScore * 100).toFixed(0)}%</Text>
          </View>
        </View>

        {/* Embedding Analytics Card */}
        <View style={styles.hudCard}>
          <Text style={styles.hudCardLabel}>128-D FACE EMBEDDING</Text>
          <Text style={styles.embeddingText} numberOfLines={1}>
            {embeddingPreview}
          </Text>
        </View>

        {/* Pipeline Telemetry Card */}
        <View style={styles.hudCard}>
          <Text style={styles.hudCardLabel}>PIPELINE TELEMETRY</Text>
          <View style={styles.telemetryRow}>
            <Text style={styles.telemetryText}>Stage 1: BlazeFace (256x256)</Text>
            <Text style={styles.telemetryStatus}>Active</Text>
          </View>
          <View style={styles.telemetryRow}>
            <Text style={styles.telemetryText}>Stage 2: MobileFaceNet INT8</Text>
            <Text style={styles.telemetryStatus}>Active</Text>
          </View>
          <View style={styles.telemetryRow}>
            <Text style={styles.telemetryText}>Processor Frame Rate</Text>
            <Text style={styles.telemetryStatus}>{fps} FPS</Text>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0f14',
  },
  center: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 20,
    textAlign: 'center',
  },
  subtext: {
    fontSize: 14,
    color: '#a0aec0',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 30,
    maxWidth: 400,
  },
  button: {
    backgroundColor: '#3b82f6',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 30,
    elevation: 3,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  // Bounding Box Overlays
  faceBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 8,
    borderStyle: 'dashed',
    zIndex: 10,
  },
  corner: {
    position: 'absolute',
    width: 16,
    height: 16,
    borderWidth: 3,
  },
  topLeft: {
    top: -2,
    left: -2,
    borderRightWidth: 0,
    borderBottomWidth: 0,
    borderTopLeftRadius: 6,
  },
  topRight: {
    top: -2,
    right: -2,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
    borderTopRightRadius: 6,
  },
  bottomLeft: {
    bottom: -2,
    left: -2,
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomLeftRadius: 6,
  },
  bottomRight: {
    bottom: -2,
    right: -2,
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderBottomRightRadius: 6,
  },
  // Side HUD Panel (Landscape layout optimization)
  hudPanel: {
    position: 'absolute',
    right: 20,
    top: 20,
    bottom: 20,
    width: 320,
    backgroundColor: 'rgba(15, 23, 42, 0.75)', // Glassmorphism dark slate
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.44,
    shadowRadius: 10.32,
    elevation: 16,
    zIndex: 20,
    justifyContent: 'space-between',
  },
  hudHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  hudTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    letterSpacing: 0.5,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  pulseDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#00ffaa',
    marginRight: 6,
  },
  liveText: {
    color: '#a0aec0',
    fontSize: 10,
    fontWeight: '600',
  },
  // HUD Cards (Glassmorphism layout)
  hudCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  hudCardLabel: {
    fontSize: 9,
    fontWeight: '700',
    color: '#718096',
    letterSpacing: 1,
    marginBottom: 4,
  },
  hudCardValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  // Dynamic Badges
  noFaceBadge: {
    borderLeftWidth: 4,
    borderLeftColor: '#718096',
  },
  verifiedBadge: {
    borderLeftWidth: 4,
    borderLeftColor: '#00ffaa',
    backgroundColor: 'rgba(0, 255, 170, 0.05)',
  },
  spoofBadge: {
    borderLeftWidth: 4,
    borderLeftColor: '#ff3b30',
    backgroundColor: 'rgba(255, 59, 48, 0.05)',
  },
  // Liveness Analytics
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  progressBarBg: {
    flex: 1,
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 4,
    overflow: 'hidden',
    marginRight: 10,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#ffffff',
    width: 32,
    textAlign: 'right',
  },
  // Embeddings Display
  embeddingText: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: '#38bdf8', // Light blue
    marginTop: 2,
  },
  // Telemetry Rows
  telemetryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  telemetryText: {
    fontSize: 11,
    color: '#a0aec0',
  },
  telemetryStatus: {
    fontSize: 11,
    fontWeight: '600',
    color: '#38bdf8',
  },
  // Welcome Screen (Landscape splits)
  welcomeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  welcomeLeft: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  welcomeRight: {
    flex: 1.2,
    justifyContent: 'center',
    paddingLeft: 24,
  },
  logoContainer: {
    marginBottom: 20,
  },
  logoRingOuter: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 2,
    borderColor: '#38bdf8',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoRingInner: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(56, 189, 248, 0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoIcon: {
    fontSize: 36,
  },
  welcomeTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    letterSpacing: 1,
  },
  welcomeSub: {
    fontSize: 13,
    color: '#a0aec0',
    textAlign: 'center',
    marginTop: 8,
    maxWidth: 260,
  },
  healthCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.06)',
    marginBottom: 20,
  },
  healthTitle: {
    fontSize: 10,
    fontWeight: '700',
    color: '#718096',
    letterSpacing: 1.5,
    marginBottom: 10,
  },
  healthRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 4,
  },
  healthLabel: {
    color: '#cbd5e0',
    fontSize: 12,
  },
  healthVal: {
    fontWeight: '600',
    fontSize: 12,
  },
  greenText: {
    color: '#00ffaa',
  },
  yellowText: {
    color: '#ecc94b',
  },
  redText: {
    color: '#ff3b30',
  },
  cyanText: {
    color: '#38bdf8',
  },
  grayText: {
    color: '#a0aec0',
  },
  scanStartBtn: {
    backgroundColor: '#38bdf8',
    borderRadius: 30,
    paddingVertical: 15,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#38bdf8',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 8,
  },
  scanStartBtnText: {
    color: '#0c0f14',
    fontSize: 14,
    fontWeight: 'bold',
    letterSpacing: 1,
  },
  backButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
  },
  backButtonText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '600',
  },
  backBtn: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
});
