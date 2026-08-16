"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DissertationDay } from "./dissertationData";

type Props = { data: DissertationDay[] };

const AXIS = "#71717a";
const GRID = "rgba(113, 113, 122, 0.15)";
const PRICE = "#f59e0b";
const MODEL_A = "#94a3b8";
const MODEL_B = "#10b981";
const MODEL_C = "#a78bfa";

const shortDate = (iso: string) => {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { month: "short", day: "numeric" });
};

const usd = (n: number) =>
  n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });

const tooltipStyle = {
  background: "rgba(24, 24, 27, 0.95)",
  border: "1px solid rgba(113, 113, 122, 0.4)",
  borderRadius: "0.5rem",
  color: "#fafafa",
  fontSize: "0.8rem",
};

export function PriceChart({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={PRICE} stopOpacity={0.35} />
            <stop offset="100%" stopColor={PRICE} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={shortDate}
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          minTickGap={32}
        />
        <YAxis
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          width={48}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={(v) => shortDate(v as string)}
          formatter={(v) => [usd(Number(v)), "Close"] as [string, string]}
        />
        <Area
          type="monotone"
          dataKey="close"
          stroke={PRICE}
          strokeWidth={2}
          fill="url(#priceFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function SentimentChart({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={shortDate}
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          minTickGap={32}
        />
        <YAxis
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          domain={[-0.4, 0.6]}
          width={48}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={(v) => shortDate(v as string)}
          formatter={(v, name) =>
            [Number(v).toFixed(3), String(name)] as [string, string]
          }
        />
        <Line
          type="monotone"
          dataKey="sentimentA"
          name="A · baseline VADER"
          stroke={MODEL_A}
          strokeWidth={1.5}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="sentimentB"
          name="B · crypto lexicon"
          stroke={MODEL_B}
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="sentimentC"
          name="C · intensity-weighted"
          stroke={MODEL_C}
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function TweetVolumeChart({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={shortDate}
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          minTickGap={32}
        />
        <YAxis
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          width={48}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={(v) => shortDate(v as string)}
          formatter={(v) =>
            [Number(v).toLocaleString(), "Tweets"] as [string, string]
          }
        />
        <Bar dataKey="tweetCount" fill={MODEL_B} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
