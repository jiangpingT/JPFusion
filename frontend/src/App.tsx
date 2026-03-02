/**
 * App.tsx — FusionLab 主布局
 *
 * 两栏结构：
 *   左  (260px)  — ParameterPanel  参数配置 + 推理控制
 *   右  (flex)   — PlasmaVisualization  等离子体可视化（全屏）
 */

import React, { useState, useCallback } from 'react';
import { ParameterPanel }       from './components/ParameterPanel';
import { PlasmaVisualization }  from './components/PlasmaVisualization';
import { RLDashboard }          from './components/RLDashboard';
import type { InferenceResult } from './api/client';

type Tab = 'plasma' | 'rl';

// ─── 主组件 ──────────────────────────────────────────────────────────────────
export default function App() {
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('rl');

  const handleInferenceResult = useCallback((result: unknown) => {
    setInferenceResult(result as InferenceResult);
  }, []);

  return (
    <div style={styles.root}>
      {/* 顶部导航栏 */}
      <div style={styles.navbar}>
        <div style={styles.navLeft}>
          <span style={styles.logo}>⚛ FusionLab</span>
          <span style={styles.navSub}>可控核聚变 AI 平台</span>
        </div>
        <div style={styles.navTabs}>
          <button
            onClick={() => setActiveTab('plasma')}
            style={{
              ...styles.tabBtn,
              ...(activeTab === 'plasma' ? styles.tabActive : {}),
            }}
          >
            等离子体推理
          </button>
          <button
            onClick={() => setActiveTab('rl')}
            style={{
              ...styles.tabBtn,
              ...(activeTab === 'rl' ? styles.tabActive : {}),
            }}
          >
            FusionRL 控制
          </button>
        </div>
        <div style={styles.navRight}>
          <span style={styles.navVersion}>FusionRL · 四阶段</span>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            style={styles.navLink}
          >
            API Docs
          </a>
        </div>
      </div>

      {/* 主体内容（Tab 切换） */}
      {activeTab === 'plasma' ? (
        <div style={styles.mainLayout}>
          {/* 左栏：参数配置 + 推理控制 */}
          <div style={styles.leftPanel}>
            <ParameterPanel onInferenceResult={handleInferenceResult} />
          </div>
          {/* 右栏：等离子体可视化 */}
          <div style={styles.rightPanel}>
            <PlasmaVisualization result={inferenceResult} />
          </div>
        </div>
      ) : (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <RLDashboard />
        </div>
      )}
    </div>
  );
}

// ─── 样式 ────────────────────────────────────────────────────────────────────
const styles: Record<string, React.CSSProperties> = {
  root: {
    display:        'flex',
    flexDirection:  'column',
    width:          '100vw',
    height:         '100vh',
    background:     '#020817',
    color:          '#e2e8f0',
    fontFamily:     "'Inter', 'SF Pro Display', -apple-system, sans-serif",
    overflow:       'hidden',
  },
  navbar: {
    display:        'flex',
    justifyContent: 'space-between',
    alignItems:     'center',
    padding:        '0 20px',
    height:         '48px',
    background:     '#0a0f1e',
    borderBottom:   '1px solid #1e293b',
    flexShrink:     0,
  },
  navLeft: {
    display:     'flex',
    alignItems:  'center',
    gap:         '12px',
  },
  logo: {
    fontSize:    '18px',
    fontWeight:  800,
    background:  'linear-gradient(135deg, #38bdf8, #818cf8, #f97316)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor:  'transparent',
    letterSpacing: '0.3px',
  },
  navSub: {
    fontSize:    '12px',
    color:       '#475569',
    borderLeft:  '1px solid #1e293b',
    paddingLeft: '12px',
  },
  navTabs: {
    display:    'flex',
    gap:        '4px',
  },
  tabBtn: {
    fontSize:     '12px',
    padding:      '4px 12px',
    borderRadius: '4px',
    border:       'none',
    background:   'transparent',
    color:        '#475569',
    cursor:       'pointer',
  } as React.CSSProperties,
  tabActive: {
    background: '#1e293b',
    color:      '#38bdf8',
  } as React.CSSProperties,
  navRight: {
    display:    'flex',
    alignItems: 'center',
    gap:        '16px',
  },
  navVersion: {
    fontSize:    '11px',
    color:       '#334155',
    padding:     '3px 8px',
    background:  '#1e293b',
    borderRadius: '4px',
  },
  navLink: {
    fontSize:   '11px',
    color:      '#38bdf8',
    textDecoration: 'none',
  },
  mainLayout: {
    display:   'flex',
    flex:      1,
    overflow:  'hidden',
  },
  leftPanel: {
    width:      '260px',
    flexShrink: 0,
    borderRight: '1px solid #1e293b',
    overflowY:  'auto',
  },
  rightPanel: {
    flex:      1,
    minWidth:  0,
    overflowY: 'auto',
  },
  rlPlaceholder: {
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'center',
    justifyContent: 'center',
    height:         '100%',
    color:          '#334155',
    padding:        '40px',
    textAlign:      'center',
  },
};
