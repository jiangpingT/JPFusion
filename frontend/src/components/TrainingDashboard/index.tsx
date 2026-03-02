/**
 * TrainingDashboard — 中间训练监控面板
 *
 * 包含：
 *   - Loss 曲线（train / val，Recharts 折线图）
 *   - Epoch 进度条
 *   - 实时指标卡片（MSE / MAE / LR / 用时）
 *   - WebSocket 连接状态
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { useWebSocket } from '../../hooks/useWebSocket';
import { WS_URL, fusionApi } from '../../api/client';

// ─── 类型 ────────────────────────────────────────────────────────────────────
interface EpochMetrics {
  epoch:        number;
  total_epochs: number;
  train_loss:   number;
  val_loss:     number;
  train_mae:    number;
  val_mae:      number;
  pde_loss:     number;
  lr:           number;
  elapsed_sec:  number;
  status:       string;
}

interface Props {
  onStatusChange?: (status: string) => void;
}

// 最多在图表上展示 200 个点（抽稀）
const MAX_CHART_POINTS = 200;

function downsample(data: EpochMetrics[], maxLen: number): EpochMetrics[] {
  if (data.length <= maxLen) return data;
  const step = Math.ceil(data.length / maxLen);
  return data.filter((_, i) => i % step === 0 || i === data.length - 1);
}

// ─── 组件 ────────────────────────────────────────────────────────────────────
export function TrainingDashboard({ onStatusChange }: Props) {
  const [history, setHistory]   = useState<EpochMetrics[]>([]);
  const [latest,  setLatest]    = useState<EpochMetrics | null>(null);
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  // 页面加载时从 REST API 拉取历史数据（刷新页面后曲线不丢失）
  useEffect(() => {
    fusionApi.getTrainingHistory().then((data: { history: EpochMetrics[]; status: string }) => {
      if (data.history && data.history.length > 0) {
        setHistory(data.history);
        setLatest(data.history[data.history.length - 1]);
        onStatusChange?.(data.status);
      }
    }).catch(() => { /* 后端未启动时忽略 */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onMessage = useCallback((data: Record<string, unknown>) => {
    const type = data.type as string;

    if (type === 'init') {
      // 页面刷新后恢复历史数据
      const hist = (data.history as EpochMetrics[]) || [];
      setHistory(hist);
      if (hist.length > 0) setLatest(hist[hist.length - 1]);
      onStatusChange?.(data.status as string);
      return;
    }

    if (type === 'training_progress') {
      const metrics = data as unknown as EpochMetrics;
      setLatest(metrics);
      onStatusChange?.(metrics.status);

      if (metrics.status === 'training') {
        setHistory(prev => {
          // 避免重复添加同一 epoch
          const last = prev[prev.length - 1];
          if (last?.epoch === metrics.epoch) return prev;
          return [...prev, metrics];
        });
      }
    }
  }, [onStatusChange]);

  useWebSocket({
    url:       WS_URL,
    onMessage,
    onOpen:    () => setWsStatus('connected'),
    onClose:   () => setWsStatus('disconnected'),
  });

  const chartData   = downsample(history, MAX_CHART_POINTS);
  const progress    = latest ? (latest.epoch / latest.total_epochs) * 100 : 0;
  const isTraining  = latest?.status === 'training';
  const isCompleted = latest?.status === 'completed';

  const wsColor = wsStatus === 'connected' ? '#4ade80' : wsStatus === 'connecting' ? '#facc15' : '#f87171';

  return (
    <div style={styles.panel}>
      {/* 标题 + WS 状态 */}
      <div style={styles.header}>
        <span style={styles.title}>📊 训练监控</span>
        <div style={styles.wsBadge}>
          <span style={{ ...styles.dot, background: wsColor }} />
          <span style={styles.wsText}>
            {wsStatus === 'connected' ? 'WS 已连接' : wsStatus === 'connecting' ? '连接中...' : '断线重连中'}
          </span>
        </div>
      </div>

      {/* Epoch 进度条 */}
      {(isTraining || isCompleted) && latest && (
        <div style={styles.progressBlock}>
          <div style={styles.progressHeader}>
            <span>Epoch {latest.epoch} / {latest.total_epochs}</span>
            <span>{isCompleted ? '✓ 训练完成' : `${progress.toFixed(0)}%`}</span>
          </div>
          <div style={styles.progressBg}>
            <div style={{ ...styles.progressFill, width: `${progress}%`, background: isCompleted ? '#4ade80' : '#38bdf8' }} />
          </div>
        </div>
      )}

      {/* 指标卡片 */}
      {latest && (
        <div style={styles.metricGrid}>
          {[
            { label: 'train_loss', name: '训练损失',  value: latest.train_loss?.toFixed(5), color: '#38bdf8' },
            { label: 'val_loss',   name: '验证损失',  value: latest.val_loss?.toFixed(5),   color: '#818cf8' },
            { label: 'train_mae',  name: '训练 MAE',  value: latest.train_mae?.toFixed(5),  color: '#34d399' },
            { label: 'val_mae',    name: '验证 MAE',  value: latest.val_mae?.toFixed(5),    color: '#fb923c' },
            { label: 'pde_loss',   name: 'PINN 物理',  value: latest.pde_loss != null ? latest.pde_loss.toFixed(5) : '—', color: '#f59e0b' },
            { label: 'lr',         name: '学习率',    value: latest.lr?.toExponential(2),   color: '#a78bfa' },
            { label: 'elapsed',    name: '已用时',    value: `${latest.elapsed_sec?.toFixed(0)}s`, color: '#94a3b8' },
          ].map(m => (
            <div key={m.label} style={styles.metricCard}>
              <div style={{ ...styles.metricValue, color: m.color }}>{m.value ?? '—'}</div>
              <div style={styles.metricLabel}>{m.label}</div>
              <div style={styles.metricCn}>{m.name}</div>
            </div>
          ))}
        </div>
      )}

      {/* Loss 曲线 */}
      <div style={styles.chartContainer}>
        <div style={styles.chartTitle}>Loss 曲线（MSE）</div>
        {chartData.length === 0 ? (
          <div style={styles.noData}>
            {wsStatus === 'connected'
              ? '等待训练开始...\n点击左侧「开始训练」按钮'
              : '正在连接后端服务...'}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="epoch"
                stroke="#475569"
                tick={{ fontSize: 10, fill: '#64748b' }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 10 }}
              />
              <YAxis
                stroke="#475569"
                tick={{ fontSize: 10, fill: '#64748b' }}
                tickFormatter={(v: number) => v.toExponential(1)}
                scale="log"
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', fontSize: '11px' }}
                labelStyle={{ color: '#94a3b8' }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(val: any) => [typeof val === 'number' ? val.toFixed(6) : String(val), '']}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', color: '#64748b' }}
              />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke="#38bdf8"
                dot={false}
                strokeWidth={2}
                name="训练 Loss"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#818cf8"
                dot={false}
                strokeWidth={2}
                name="验证 Loss"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* MAE 曲线 */}
      {chartData.length > 0 && (
        <div style={styles.chartContainer}>
          <div style={styles.chartTitle}>MAE 曲线（平均绝对误差）</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} />
              <YAxis stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={(v: number) => v.toFixed(3)} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', fontSize: '11px' }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(val: any) => [typeof val === 'number' ? val.toFixed(5) : String(val), '']}
              />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#64748b' }} />
              <Line type="monotone" dataKey="train_mae" stroke="#34d399" dot={false} strokeWidth={1.5} name="训练 MAE" isAnimationActive={false} />
              <Line type="monotone" dataKey="val_mae"   stroke="#fb923c" dot={false} strokeWidth={1.5} name="验证 MAE"  isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 名词备注 */}
      <div style={styles.glossary}>
        <div style={styles.glossaryTitle}>【名词备注】</div>
        <div style={styles.glossaryItem}><b>MSE Loss</b>（均方误差）— 预测值与真实值差的平方均值，越小越好，对大误差惩罚更重</div>
        <div style={styles.glossaryItem}><b>MAE</b>（平均绝对误差）— 预测误差的绝对值均值，物理意义更直观</div>
        <div style={styles.glossaryItem}><b>Val Loss</b>（验证损失）— 在未见过的数据上的误差，衡量泛化能力</div>
        <div style={styles.glossaryItem}><b>LR</b>（学习率）— 每步参数更新的步长，使用余弦退火，从 1e-3 缓降到 1e-5</div>
        <div style={styles.glossaryItem}><b>PINN 物理</b>（物理约束损失）— PDE 残差 + 单调性 + 对称性 + 边界条件四项之和，越小代表模型越符合热传导物理规律</div>
      </div>
    </div>
  );
}

// ─── 样式 ────────────────────────────────────────────────────────────────────
const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: '100%',
    height: '100%',
    overflowY: 'auto',
    background: '#0a0f1e',
    padding: '16px',
    color: '#e2e8f0',
    fontFamily: "'Inter', 'SF Pro Display', sans-serif",
    fontSize: '13px',
    boxSizing: 'border-box',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 700,
    color: '#818cf8',
  },
  wsBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    fontSize: '11px',
    color: '#64748b',
  },
  dot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
  },
  wsText: {},
  progressBlock: {
    marginBottom: '16px',
  },
  progressHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '11px',
    color: '#64748b',
    marginBottom: '4px',
  },
  progressBg: {
    height: '6px',
    background: '#1e293b',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  },
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '6px',
    marginBottom: '16px',
  },
  metricCard: {
    background: '#1e293b',
    borderRadius: '8px',
    padding: '10px 8px',
    textAlign: 'center',
  },
  metricValue: {
    fontSize: '14px',
    fontWeight: 700,
    fontFamily: 'monospace',
    marginBottom: '2px',
  },
  metricLabel: {
    fontSize: '9px',
    color: '#475569',
    fontFamily: 'monospace',
  },
  metricCn: {
    fontSize: '9px',
    color: '#334155',
    marginTop: '1px',
  },
  chartContainer: {
    background: '#0f172a',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px',
  },
  chartTitle: {
    fontSize: '11px',
    color: '#475569',
    marginBottom: '8px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  noData: {
    height: '200px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#334155',
    fontSize: '13px',
    textAlign: 'center',
    whiteSpace: 'pre-line',
    lineHeight: '1.8',
  },
  glossary: {
    padding: '12px',
    background: '#1e293b',
    borderRadius: '8px',
    borderLeft: '3px solid #818cf8',
  },
  glossaryTitle: {
    fontSize: '11px',
    fontWeight: 700,
    color: '#818cf8',
    marginBottom: '8px',
  },
  glossaryItem: {
    fontSize: '10px',
    color: '#64748b',
    marginBottom: '4px',
    lineHeight: '1.5',
  },
};
