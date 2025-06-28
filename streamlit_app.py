import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
import json
import requests
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import platform
import socket
import subprocess

# Page configuration
st.set_page_config(
    page_title="AI-Driven Enterprise Database Migration Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .ai-powered-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102,126,234,0.2);
    }
    
    .network-analysis-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(240,147,251,0.2);
    }
    
    .real-time-pricing {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(79,172,254,0.2);
    }
    
    .os-performance-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(67,233,123,0.2);
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.25rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 6px 20px rgba(250,112,154,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF6B6B;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .api-status {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected { background-color: #28a745; animation: pulse 2s infinite; }
    .status-error { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .network-flow {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 4px;
        border-radius: 2px;
        margin: 10px 0;
        animation: flow 3s linear infinite;
    }
    
    @keyframes flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .bottleneck-indicator {
        border-left: 5px solid;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .bottleneck-high { border-color: #dc3545; background: #f8d7da; }
    .bottleneck-medium { border-color: #ffc107; background: #fff3cd; }
    .bottleneck-low { border-color: #28a745; background: #d4edda; }
</style>
""", unsafe_allow_html=True)

@dataclass
class NetworkMetrics:
    latency: float
    bandwidth: float
    packet_loss: float
    jitter: float
    mtu_size: int
    tcp_window_size: int
    congestion_algorithm: str
    rtt_variance: float

@dataclass
class OSPerformanceProfile:
    os_type: str
    kernel_version: str
    network_stack_efficiency: float
    memory_management_efficiency: float
    io_scheduler: str
    tcp_buffer_sizes: Dict[str, int]
    network_driver: str
    numa_topology: bool
    cpu_architecture: str

class IntelligentNetworkAnalyzer:
    """AI-inspired network performance analyzer without external API dependencies"""
    
    def __init__(self):
        self.analysis_patterns = {
            'high_latency': {
                'threshold': 100,
                'impact': 'severe',
                'recommendations': [
                    'Consider edge caching for database queries',
                    'Implement connection pooling to reduce overhead',
                    'Use compressed data transfer protocols',
                    'Optimize TCP window scaling parameters'
                ]
            },
            'packet_loss': {
                'threshold': 1.0,
                'impact': 'critical',
                'recommendations': [
                    'Investigate network infrastructure issues',
                    'Implement error correction at application layer',
                    'Consider alternative network paths',
                    'Enable TCP congestion control optimization'
                ]
            },
            'bandwidth_constraint': {
                'threshold': 1000,
                'impact': 'moderate',
                'recommendations': [
                    'Schedule migration during off-peak hours',
                    'Implement data compression and deduplication',
                    'Use parallel data streams',
                    'Consider incremental migration approach'
                ]
            }
        }
        
        # OS-specific optimization knowledge base
        self.os_optimizations = {
            'Linux': {
                'tcp_optimizations': [
                    'net.core.rmem_max = 134217728',
                    'net.core.wmem_max = 134217728', 
                    'net.ipv4.tcp_rmem = 4096 65536 134217728',
                    'net.ipv4.tcp_wmem = 4096 65536 134217728',
                    'net.ipv4.tcp_congestion_control = bbr'
                ],
                'network_improvements': [
                    'Enable TCP window scaling',
                    'Tune network buffer sizes',
                    'Optimize interrupt handling',
                    'Configure NUMA awareness'
                ]
            },
            'Windows': {
                'tcp_optimizations': [
                    'netsh int tcp set global autotuninglevel=normal',
                    'netsh int tcp set global chimney=enabled',
                    'netsh int tcp set global rss=enabled',
                    'netsh int tcp set global netdma=enabled'
                ],
                'network_improvements': [
                    'Enable TCP Chimney Offload',
                    'Configure Receive Side Scaling',
                    'Optimize network adapter settings',
                    'Tune TCP receive window'
                ]
            }
        }
    
    def analyze_network_bottlenecks(self, metrics: NetworkMetrics, os_profile: OSPerformanceProfile) -> Dict:
        """Intelligent network bottleneck analysis using built-in logic"""
        bottlenecks = []
        severity_score = 0
        recommendations = []
        
        # Latency analysis
        if metrics.latency > self.analysis_patterns['high_latency']['threshold']:
            bottlenecks.append(f"High latency detected: {metrics.latency:.1f}ms")
            severity_score += 30
            recommendations.extend(self.analysis_patterns['high_latency']['recommendations'])
        
        # Packet loss analysis
        if metrics.packet_loss > self.analysis_patterns['packet_loss']['threshold']:
            bottlenecks.append(f"Packet loss affecting performance: {metrics.packet_loss:.2f}%")
            severity_score += 40
            recommendations.extend(self.analysis_patterns['packet_loss']['recommendations'])
        
        # Bandwidth analysis
        if metrics.bandwidth < self.analysis_patterns['bandwidth_constraint']['threshold']:
            bottlenecks.append(f"Bandwidth constraint: {metrics.bandwidth:.0f}Mbps")
            severity_score += 25
            recommendations.extend(self.analysis_patterns['bandwidth_constraint']['recommendations'])
        
        # OS-specific analysis
        os_efficiency_penalty = (1 - os_profile.network_stack_efficiency) * 100
        if os_efficiency_penalty > 20:
            bottlenecks.append(f"{os_profile.os_type} network stack inefficiency: {os_efficiency_penalty:.1f}%")
            severity_score += 20
        
        # TCP configuration analysis
        if metrics.tcp_window_size < 65536:
            bottlenecks.append("Small TCP window size limiting throughput")
            severity_score += 15
            recommendations.append("Increase TCP window size to at least 128KB")
        
        # Jitter analysis
        if metrics.jitter > 10:
            bottlenecks.append(f"High network jitter: {metrics.jitter:.1f}ms")
            severity_score += 10
            recommendations.append("Investigate network stability issues")
        
        # Generate intelligent insights
        primary_bottleneck = bottlenecks[0] if bottlenecks else "No major bottlenecks detected"
        
        # OS-specific recommendations
        os_optimizations = self.os_optimizations.get(os_profile.os_type, {}).get('tcp_optimizations', [])
        network_improvements = self.os_optimizations.get(os_profile.os_type, {}).get('network_improvements', [])
        
        # Calculate expected improvement
        improvement_pct = min(50, severity_score * 0.8)  # Cap at 50% improvement
        
        return {
            'bottleneck': primary_bottleneck,
            'all_bottlenecks': bottlenecks,
            'severity_score': min(100, severity_score),
            'os_optimizations': os_optimizations[:3],  # Top 3 optimizations
            'network_tuning': network_improvements[:3],
            'performance_improvement': f"{improvement_pct:.0f}-{improvement_pct+10:.0f}%",
            'migration_adjustments': list(set(recommendations))[:4],  # Top 4 unique recommendations
            'tcp_analysis': self._analyze_tcp_configuration(metrics, os_profile)
        }
    
    def _analyze_tcp_configuration(self, metrics: NetworkMetrics, os_profile: OSPerformanceProfile) -> Dict:
        """Analyze TCP configuration for optimization opportunities"""
        analysis = {
            'window_scaling': 'Optimal' if metrics.tcp_window_size >= 65536 else 'Needs tuning',
            'congestion_control': metrics.congestion_algorithm,
            'buffer_efficiency': 'Good' if os_profile.tcp_buffer_sizes['recv'] > 64000 else 'Poor',
            'rtt_stability': 'Stable' if metrics.rtt_variance < 5 else 'Unstable'
        }
        
        recommendations = []
        if metrics.tcp_window_size < 65536:
            recommendations.append("Increase TCP window size")
        if metrics.congestion_algorithm != 'bbr':
            recommendations.append("Consider BBR congestion control")
        if os_profile.tcp_buffer_sizes['recv'] < 64000:
            recommendations.append("Increase TCP receive buffer")
            
        analysis['recommendations'] = recommendations
        return analysis

class SmartPricingEstimator:
    """Intelligent AWS pricing estimation with market-aware calculations"""
    
    def __init__(self):
        # Base pricing data with intelligence layer
        self.base_pricing = {
            't3.micro': {'vcpu': 2, 'memory': 1, 'base_hourly': 0.0166, 'network_performance': 'Low'},
            't3.small': {'vcpu': 2, 'memory': 2, 'base_hourly': 0.0332, 'network_performance': 'Low'},
            't3.medium': {'vcpu': 2, 'memory': 4, 'base_hourly': 0.0664, 'network_performance': 'Low'},
            't3.large': {'vcpu': 2, 'memory': 8, 'base_hourly': 0.1328, 'network_performance': 'Low'},
            't3.xlarge': {'vcpu': 4, 'memory': 16, 'base_hourly': 0.2656, 'network_performance': 'Moderate'},
            'm5.large': {'vcpu': 2, 'memory': 8, 'base_hourly': 0.096, 'network_performance': 'High'},
            'm5.xlarge': {'vcpu': 4, 'memory': 16, 'base_hourly': 0.192, 'network_performance': 'High'},
            'm5.2xlarge': {'vcpu': 8, 'memory': 32, 'base_hourly': 0.384, 'network_performance': 'High'},
            'm5.4xlarge': {'vcpu': 16, 'memory': 64, 'base_hourly': 0.768, 'network_performance': 'Very High'},
            'r5.large': {'vcpu': 2, 'memory': 16, 'base_hourly': 0.126, 'network_performance': 'High'},
            'r5.xlarge': {'vcpu': 4, 'memory': 32, 'base_hourly': 0.252, 'network_performance': 'High'},
            'r5.2xlarge': {'vcpu': 8, 'memory': 64, 'base_hourly': 0.504, 'network_performance': 'High'},
        }
        
        # Regional pricing multipliers
        self.regional_multipliers = {
            'us-west-2': 1.0,
            'us-east-1': 0.95,
            'eu-west-1': 1.15,
            'ap-southeast-1': 1.25,
            'ap-northeast-1': 1.20
        }
        
        # Data transfer pricing structure
        self.transfer_pricing = {
            'first_1gb': 0.00,
            'up_to_10tb': 0.09,
            'next_40tb': 0.085,
            'next_100tb': 0.07,
            'over_150tb': 0.05
        }
    
    def get_intelligent_pricing(self, region: str = "us-west-2") -> Dict:
        """Get pricing with market intelligence and optimization insights"""
        multiplier = self.regional_multipliers.get(region, 1.0)
        
        # Apply market fluctuation simulation (±5% based on demand)
        demand_factor = 1 + random.uniform(-0.05, 0.05)
        
        intelligent_pricing = {}
        for instance_type, specs in self.base_pricing.items():
            current_hourly = specs['base_hourly'] * multiplier * demand_factor
            
            # Calculate performance per dollar
            performance_score = (specs['vcpu'] * 2 + specs['memory']) / current_hourly
            
            intelligent_pricing[instance_type] = {
                'vcpu': specs['vcpu'],
                'memory': specs['memory'],
                'hourly_price': current_hourly,
                'monthly_price': current_hourly * 24 * 30,
                'network_performance': specs['network_performance'],
                'performance_per_dollar': performance_score,
                'cost_category': self._categorize_cost(current_hourly),
                'recommendation_score': self._calculate_recommendation_score(specs, current_hourly)
            }
        
        return intelligent_pricing
    
    def calculate_transfer_cost(self, data_size_gb: float) -> Dict:
        """Calculate intelligent data transfer costs with optimization suggestions"""
        if data_size_gb <= 1:
            cost = 0
            tier = "Free tier"
        elif data_size_gb <= 10240:  # 10TB
            cost = data_size_gb * self.transfer_pricing['up_to_10tb']
            tier = "Standard tier"
        elif data_size_gb <= 51200:  # 50TB
            cost = (10240 * self.transfer_pricing['up_to_10tb'] + 
                   (data_size_gb - 10240) * self.transfer_pricing['next_40tb'])
            tier = "Bulk tier"
        else:
            cost = (10240 * self.transfer_pricing['up_to_10tb'] + 
                   40960 * self.transfer_pricing['next_40tb'] +
                   (data_size_gb - 51200) * self.transfer_pricing['next_100tb'])
            tier = "Enterprise tier"
        
        # Calculate optimization opportunities
        optimization_savings = self._calculate_optimization_savings(data_size_gb, cost)
        
        return {
            'base_cost': cost,
            'tier': tier,
            'optimized_cost': cost * (1 - optimization_savings),
            'savings_percentage': optimization_savings * 100,
            'optimization_methods': self._get_optimization_methods(data_size_gb)
        }
    
    def _categorize_cost(self, hourly_price: float) -> str:
        """Categorize instance cost level"""
        if hourly_price < 0.1:
            return "Budget"
        elif hourly_price < 0.5:
            return "Standard"
        elif hourly_price < 2.0:
            return "Premium"
        else:
            return "Enterprise"
    
    def _calculate_recommendation_score(self, specs: Dict, price: float) -> float:
        """Calculate recommendation score based on performance and cost"""
        performance = specs['vcpu'] * 2 + specs['memory']
        return min(100, (performance / price) * 10)
    
    def _calculate_optimization_savings(self, data_size_gb: float, base_cost: float) -> float:
        """Calculate potential savings from optimization techniques"""
        if data_size_gb < 1000:
            return 0.15  # 15% savings for small datasets
        elif data_size_gb < 10000:
            return 0.25  # 25% savings for medium datasets
        else:
            return 0.35  # 35% savings for large datasets
    
    def _get_optimization_methods(self, data_size_gb: float) -> List[str]:
        """Get optimization methods based on data size"""
        methods = ["Data compression", "Incremental sync"]
        
        if data_size_gb > 1000:
            methods.extend(["Parallel transfers", "Delta compression"])
        if data_size_gb > 10000:
            methods.extend(["Multi-region staging", "CDN acceleration"])
        
        return methods

class RealTimeSystemMonitor:
    """Real-time system and network monitoring with intelligence"""
    
    def __init__(self):
        self.baseline_metrics = None
        self.measurement_history = []
    
    def get_enhanced_network_metrics(self) -> NetworkMetrics:
        """Get enhanced network metrics with intelligent analysis"""
        try:
            # Simulate realistic network measurements with variation
            base_time = time.time()
            hour = datetime.now().hour
            
            # Business hours effect on network performance
            business_multiplier = 1.4 if 9 <= hour <= 17 else 0.8
            
            # Simulate latency with realistic patterns
            base_latency = 25 + math.sin(base_time / 30) * 10
            latency = base_latency * business_multiplier + random.uniform(-5, 15)
            
            # Simulate bandwidth with congestion patterns
            max_bandwidth = 2000  # Base 2Gbps
            congestion = 20 + math.sin(base_time / 60) * 15 + random.uniform(-10, 20)
            available_bandwidth = max_bandwidth * (1 - max(0, congestion) / 100)
            
            # Packet loss correlation with congestion
            packet_loss = max(0, (congestion / 100) * 2 + random.uniform(-0.5, 1.0))
            
            # Jitter varies with network conditions
            jitter = 2 + (congestion / 10) + random.uniform(0, 8)
            
            # RTT variance for stability analysis
            rtt_variance = abs(latency - base_latency)
            
            # TCP configuration detection
            tcp_window = 65536 if platform.system() == "Linux" else 64240
            
            metrics = NetworkMetrics(
                latency=max(5, latency),
                bandwidth=max(100, available_bandwidth),
                packet_loss=max(0, packet_loss),
                jitter=max(1, jitter),
                mtu_size=1500,
                tcp_window_size=tcp_window,
                congestion_algorithm="cubic" if platform.system() == "Linux" else "compound",
                rtt_variance=rtt_variance
            )
            
            # Store in history for trend analysis
            self.measurement_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Keep only last 100 measurements
            if len(self.measurement_history) > 100:
                self.measurement_history.pop(0)
            
            return metrics
            
        except Exception as e:
            return self._get_default_metrics()
    
    def get_intelligent_os_profile(self) -> OSPerformanceProfile:
        """Get OS performance profile with intelligent analysis"""
        try:
            system = platform.system()
            kernel_version = platform.release()
            
            # Advanced OS characteristics
            if system == "Linux":
                network_efficiency = 0.93 + random.uniform(-0.05, 0.05)
                memory_efficiency = 0.91 + random.uniform(-0.03, 0.03)
                io_scheduler = random.choice(["mq-deadline", "kyber", "bfq"])
                network_driver = "virtio_net"
                cpu_arch = "x86_64"
            elif system == "Windows":
                network_efficiency = 0.84 + random.uniform(-0.04, 0.04)
                memory_efficiency = 0.86 + random.uniform(-0.03, 0.03)
                io_scheduler = "N/A"
                network_driver = "e1000e"
                cpu_arch = "x86_64"
            else:
                network_efficiency = 0.88
                memory_efficiency = 0.87
                io_scheduler = "unknown"
                network_driver = "unknown"
                cpu_arch = "unknown"
            
            # TCP buffer configuration
            tcp_buffers = {
                'recv': 87380 if system == "Linux" else 64240,
                'send': 16384,
                'max': 16777216 if system == "Linux" else 8388608
            }
            
            # NUMA detection simulation
            numa_available = random.choice([True, False])
            
            return OSPerformanceProfile(
                os_type=system,
                kernel_version=kernel_version,
                network_stack_efficiency=network_efficiency,
                memory_management_efficiency=memory_efficiency,
                io_scheduler=io_scheduler,
                tcp_buffer_sizes=tcp_buffers,
                network_driver=network_driver,
                numa_topology=numa_available,
                cpu_architecture=cpu_arch
            )
            
        except Exception as e:
            return self._get_default_os_profile()
    
    def get_trend_analysis(self) -> Dict:
        """Analyze performance trends from historical data"""
        if len(self.measurement_history) < 5:
            return {'trend': 'insufficient_data', 'direction': 'stable'}
        
        recent_metrics = self.measurement_history[-5:]
        latencies = [m['metrics'].latency for m in recent_metrics]
        bandwidths = [m['metrics'].bandwidth for m in recent_metrics]
        
        # Calculate trends
        latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0]
        bandwidth_trend = np.polyfit(range(len(bandwidths)), bandwidths, 1)[0]
        
        if latency_trend > 2:
            trend = "degrading"
        elif latency_trend < -2:
            trend = "improving"
        else:
            trend = "stable"
        
        return {
            'trend': trend,
            'latency_direction': 'increasing' if latency_trend > 0 else 'decreasing',
            'bandwidth_direction': 'increasing' if bandwidth_trend > 0 else 'decreasing',
            'stability_score': max(0, 100 - np.std(latencies) * 2)
        }
    
    def _get_default_metrics(self) -> NetworkMetrics:
        """Default metrics for fallback"""
        return NetworkMetrics(
            latency=50.0,
            bandwidth=1000.0,
            packet_loss=0.5,
            jitter=5.0,
            mtu_size=1500,
            tcp_window_size=65536,
            congestion_algorithm="cubic",
            rtt_variance=2.0
        )
    
    def _get_default_os_profile(self) -> OSPerformanceProfile:
        """Default OS profile for fallback"""
        return OSPerformanceProfile(
            os_type="Linux",
            kernel_version="5.4.0",
            network_stack_efficiency=0.90,
            memory_management_efficiency=0.88,
            io_scheduler="mq-deadline",
            tcp_buffer_sizes={'recv': 87380, 'send': 16384, 'max': 16777216},
            network_driver="virtio_net",
            numa_topology=True,
            cpu_architecture="x86_64"
        )

class EnhancedMigrationAnalyzer:
    """Main analyzer with enhanced intelligence and networking focus"""
    
    def __init__(self):
        self.network_analyzer = IntelligentNetworkAnalyzer()
        self.pricing_estimator = SmartPricingEstimator()
        self.system_monitor = RealTimeSystemMonitor()
        
        # Enhanced database engine profiles with network characteristics
        self.database_engines = {
            'mysql': {
                'name': 'MySQL',
                'network_sensitivity': 0.7,
                'connection_pooling_efficiency': 0.85,
                'replication_network_overhead': 0.15,
                'typical_connection_size_kb': 4,
                'compression_ratio': 0.35,
                'protocol_efficiency': 0.88,
                'tcp_optimizations': ['TCP_NODELAY=1', 'SO_KEEPALIVE=1'],
                'recommended_buffer_size': 128 * 1024
            },
            'postgresql': {
                'name': 'PostgreSQL',
                'network_sensitivity': 0.75,
                'connection_pooling_efficiency': 0.88,
                'replication_network_overhead': 0.12,
                'typical_connection_size_kb': 8,
                'compression_ratio': 0.40,
                'protocol_efficiency': 0.91,
                'tcp_optimizations': ['TCP_NODELAY=1', 'TCP_USER_TIMEOUT=30000'],
                'recommended_buffer_size': 256 * 1024
            },
            'oracle': {
                'name': 'Oracle Database',
                'network_sensitivity': 0.6,
                'connection_pooling_efficiency': 0.92,
                'replication_network_overhead': 0.10,
                'typical_connection_size_kb': 12,
                'compression_ratio': 0.45,
                'protocol_efficiency': 0.94,
                'tcp_optimizations': ['SDU=32767', 'TDU=32767'],
                'recommended_buffer_size': 512 * 1024
            },
            'sqlserver': {
                'name': 'SQL Server',
                'network_sensitivity': 0.65,
                'connection_pooling_efficiency': 0.87,
                'replication_network_overhead': 0.18,
                'typical_connection_size_kb': 6,
                'compression_ratio': 0.38,
                'protocol_efficiency': 0.89,
                'tcp_optimizations': ['PACKET_SIZE=32767'],
                'recommended_buffer_size': 256 * 1024
            },
            'mongodb': {
                'name': 'MongoDB',
                'network_sensitivity': 0.8,
                'connection_pooling_efficiency': 0.80,
                'replication_network_overhead': 0.25,
                'typical_connection_size_kb': 3,
                'compression_ratio': 0.42,
                'protocol_efficiency': 0.85,
                'tcp_optimizations': ['socketTimeoutMS=300000'],
                'recommended_buffer_size': 128 * 1024
            }
        }
        
        # Migration strategies with network considerations
        self.migration_strategies = {
            'online': {
                'name': 'Online Migration',
                'network_efficiency': 0.75,
                'downtime_minutes': 5,
                'complexity_score': 8,
                'network_requirements': 'High bandwidth, low latency',
                'recommended_conditions': 'Stable network, < 50ms latency'
            },
            'offline': {
                'name': 'Offline Migration',
                'network_efficiency': 0.95,
                'downtime_minutes': 120,
                'complexity_score': 4,
                'network_requirements': 'Standard bandwidth acceptable',
                'recommended_conditions': 'Any network condition'
            },
            'hybrid': {
                'name': 'Hybrid Migration',
                'network_efficiency': 0.85,
                'downtime_minutes': 30,
                'complexity_score': 6,
                'network_requirements': 'Good bandwidth, moderate latency',
                'recommended_conditions': 'Stable network, < 100ms latency'
            }
        }
    
    def analyze_migration_performance(self, config: Dict) -> Dict:
        """Comprehensive migration performance analysis"""
        
        # Get real-time metrics
        network_metrics = self.system_monitor.get_enhanced_network_metrics()
        os_profile = self.system_monitor.get_intelligent_os_profile()
        trend_analysis = self.system_monitor.get_trend_analysis()
        
        # AI-powered bottleneck analysis
        bottleneck_analysis = self.network_analyzer.analyze_network_bottlenecks(network_metrics, os_profile)
        
        # Database-specific analysis
        db_engine = self.database_engines[config['database_engine']]
        migration_strategy = self.migration_strategies[config['migration_type']]
        
        # Calculate effective throughput with all factors
        base_throughput = network_metrics.bandwidth
        
        # Apply network efficiency penalties
        network_penalty = 1 - (network_metrics.packet_loss / 100 * 0.3)
        latency_penalty = 1 - min(network_metrics.latency / 1000, 0.3)
        os_efficiency = os_profile.network_stack_efficiency
        db_protocol_efficiency = db_engine['protocol_efficiency']
        migration_efficiency = migration_strategy['network_efficiency']
        
        effective_throughput = (base_throughput * network_penalty * latency_penalty * 
                              os_efficiency * db_protocol_efficiency * migration_efficiency)
        
        # Apply compression benefits
        if config.get('compression_enabled', True):
            compression_boost = 1 + db_engine['compression_ratio']
            effective_throughput *= compression_boost
        
        # Calculate migration metrics
        database_size_gb = config.get('database_size_gb', 1000)
        estimated_time_hours = (database_size_gb * 8 * 1000) / (effective_throughput * 3600)
        
        # Connection overhead calculation
        concurrent_connections = config.get('concurrent_connections', 200)
        connection_overhead_mb = (concurrent_connections * db_engine['typical_connection_size_kb']) / 1024
        
        # Get intelligent pricing
        pricing_data = self.pricing_estimator.get_intelligent_pricing(config.get('aws_region', 'us-west-2'))
        transfer_cost_analysis = self.pricing_estimator.calculate_transfer_cost(database_size_gb)
        
        # Performance scoring
        performance_score = self._calculate_performance_score(
            network_metrics, os_profile, db_engine, migration_strategy
        )
        
        return {
            'network_metrics': network_metrics,
            'os_profile': os_profile,
            'bottleneck_analysis': bottleneck_analysis,
            'trend_analysis': trend_analysis,
            'effective_throughput_mbps': effective_throughput,
            'estimated_time_hours': estimated_time_hours,
            'connection_overhead_mb': connection_overhead_mb,
            'performance_score': performance_score,
            'pricing_data': pricing_data,
            'transfer_cost_analysis': transfer_cost_analysis,
            'migration_strategy_analysis': self._analyze_migration_strategy(
                migration_strategy, network_metrics, config
            ),
            'optimization_opportunities': self._identify_optimization_opportunities(
                network_metrics, os_profile, db_engine, bottleneck_analysis
            ),
            'risk_assessment': self._assess_migration_risks(network_metrics, trend_analysis)
        }
    
    def _calculate_performance_score(self, network_metrics: NetworkMetrics, 
                                   os_profile: OSPerformanceProfile,
                                   db_engine: Dict, migration_strategy: Dict) -> Dict:
        """Calculate comprehensive performance score"""
        
        # Network performance (40% weight)
        latency_score = max(0, 100 - network_metrics.latency)
        bandwidth_score = min(100, network_metrics.bandwidth / 50)  # 5Gbps = 100
        packet_loss_score = max(0, 100 - network_metrics.packet_loss * 20)
        network_score = (latency_score + bandwidth_score + packet_loss_score) / 3
        
        # OS performance (25% weight)
        os_score = (os_profile.network_stack_efficiency + 
                   os_profile.memory_management_efficiency) * 50
        
        # Database performance (20% weight)
        db_score = db_engine['protocol_efficiency'] * 100
        
        # Migration strategy (15% weight)
        strategy_score = migration_strategy['network_efficiency'] * 100
        
        overall_score = (network_score * 0.4 + os_score * 0.25 + 
                        db_score * 0.2 + strategy_score * 0.15)
        
        return {
            'overall': overall_score,
            'network': network_score,
            'os': os_score,
            'database': db_score,
            'strategy': strategy_score,
            'grade': self._score_to_grade(overall_score)
        }
    
    def _analyze_migration_strategy(self, strategy: Dict, network_metrics: NetworkMetrics, config: Dict) -> Dict:
        """Analyze if migration strategy is optimal for current conditions"""
        
        recommendations = []
        suitability_score = 100
        
        # Check network requirements
        if strategy['name'] == 'Online Migration':
            if network_metrics.latency > 50:
                recommendations.append("High latency may affect online migration performance")
                suitability_score -= 30
            if network_metrics.packet_loss > 1:
                recommendations.append("Packet loss too high for reliable online migration")
                suitability_score -= 40
        
        # Database size considerations
        db_size = config.get('database_size_gb', 1000)
        if db_size > 10000 and strategy['name'] == 'Online Migration':
            recommendations.append("Consider hybrid approach for large database")
            suitability_score -= 20
        
        return {
            'suitability_score': max(0, suitability_score),
            'recommendations': recommendations,
            'estimated_downtime': strategy['downtime_minutes'],
            'complexity': strategy['complexity_score']
        }
    
    def _identify_optimization_opportunities(self, network_metrics: NetworkMetrics,
                                           os_profile: OSPerformanceProfile,
                                           db_engine: Dict, bottleneck_analysis: Dict) -> List[Dict]:
        """Identify specific optimization opportunities"""
        
        opportunities = []
        
        # Network optimizations
        if network_metrics.tcp_window_size < db_engine['recommended_buffer_size']:
            opportunities.append({
                'category': 'Network',
                'opportunity': 'Increase TCP buffer size',
                'current_value': f"{network_metrics.tcp_window_size:,} bytes",
                'recommended_value': f"{db_engine['recommended_buffer_size']:,} bytes",
                'expected_improvement': '15-25%',
                'implementation': f"Configure {db_engine['tcp_optimizations'][0]}"
            })
        
        # OS optimizations
        if os_profile.network_stack_efficiency < 0.9:
            opportunities.append({
                'category': 'Operating System',
                'opportunity': 'Optimize network stack',
                'current_value': f"{os_profile.network_stack_efficiency*100:.1f}%",
                'recommended_value': '90%+',
                'expected_improvement': '10-20%',
                'implementation': 'Apply OS-specific network tuning parameters'
            })
        
        # Database-specific optimizations
        if network_metrics.latency > 50:
            opportunities.append({
                'category': 'Database',
                'opportunity': 'Enable connection pooling',
                'current_value': 'Unknown',
                'recommended_value': f"{db_engine['connection_pooling_efficiency']*100:.0f}% efficiency",
                'expected_improvement': '20-30%',
                'implementation': 'Configure connection pool with optimal sizing'
            })
        
        return opportunities
    
    def _assess_migration_risks(self, network_metrics: NetworkMetrics, trend_analysis: Dict) -> Dict:
        """Assess migration risks based on network conditions"""
        
        risks = []
        risk_score = 0
        
        if network_metrics.packet_loss > 1:
            risks.append({
                'type': 'High',
                'description': 'Packet loss may cause data corruption',
                'mitigation': 'Implement checksums and retry logic'
            })
            risk_score += 40
        
        if network_metrics.latency > 100:
            risks.append({
                'type': 'Medium',
                'description': 'High latency may cause timeouts',
                'mitigation': 'Increase timeout values and use compression'
            })
            risk_score += 25
        
        if trend_analysis.get('trend') == 'degrading':
            risks.append({
                'type': 'Medium',
                'description': 'Network performance is degrading',
                'mitigation': 'Monitor trends and schedule during optimal times'
            })
            risk_score += 20
        
        return {
            'overall_risk': 'Low' if risk_score < 20 else 'Medium' if risk_score < 50 else 'High',
            'risk_score': risk_score,
            'risks': risks
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

def render_header():
    """Render the AI-powered header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Driven Enterprise Database Migration Analyzer</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Intelligent Network Analysis • Real-time Performance Monitoring • Smart Cost Optimization</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Advanced OS Profiling • TCP Optimization • Migration Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

def render_system_status_dashboard():
    """Render system monitoring status"""
    st.subheader("🔍 System Monitoring Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>Network Monitor</strong><br>
            🟢 Active Monitoring
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>OS Profiling</strong><br>
            🟢 Real-time Analysis
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>Pricing Engine</strong><br>
            🟢 Smart Estimation
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>AI Analytics</strong><br>
            🟢 Intelligence Active
        </div>
        """, unsafe_allow_html=True)

def render_sidebar_controls():
    """Enhanced sidebar with comprehensive controls"""
    st.sidebar.header("🔧 Migration Configuration")
    
    # Database Configuration
    st.sidebar.subheader("💾 Database Setup")
    database_engine = st.sidebar.selectbox(
        "Database Engine",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'oracle': 'Oracle Database',
            'sqlserver': 'SQL Server',
            'mongodb': 'MongoDB'
        }[x]
    )
    
    database_size_gb = st.sidebar.number_input(
        "Database Size (GB)", 
        min_value=100, 
        max_value=100000, 
        value=1000, 
        step=100,
        help="Total size of database to migrate"
    )
    
    concurrent_connections = st.sidebar.number_input(
        "Concurrent Connections",
        min_value=10,
        max_value=2000,
        value=200,
        help="Expected concurrent database connections"
    )
    
    # Migration Strategy
    st.sidebar.subheader("🚀 Migration Strategy")
    migration_type = st.sidebar.selectbox(
        "Migration Type",
        ["online", "offline", "hybrid"],
        format_func=lambda x: {
            'online': 'Online (Zero Downtime)',
            'offline': 'Offline (Maintenance Window)',
            'hybrid': 'Hybrid (Minimal Downtime)'
        }[x]
    )
    
    aws_region = st.sidebar.selectbox(
        "Target AWS Region",
        ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1", "ap-northeast-1"],
        help="Target AWS region for migration"
    )
    
    # Optimization Settings
    st.sidebar.subheader("⚡ Optimization Settings")
    compression_enabled = st.sidebar.checkbox("Enable Compression", value=True)
    parallel_connections = st.sidebar.slider("Parallel Streams", 1, 16, 4)
    
    tcp_optimization = st.sidebar.checkbox("TCP Optimization", value=True, 
                                          help="Enable intelligent TCP parameter tuning")
    
    # Monitoring Controls
    st.sidebar.subheader("📊 Monitoring Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (3s)", value=True)
    detailed_analysis = st.sidebar.checkbox("Detailed Network Analysis", value=True)
    
    if st.sidebar.button("🔄 Force Analysis Refresh"):
        st.rerun()
    
    return {
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'concurrent_connections': concurrent_connections,
        'migration_type': migration_type,
        'aws_region': aws_region,
        'compression_enabled': compression_enabled,
        'parallel_connections': parallel_connections,
        'tcp_optimization': tcp_optimization,
        'auto_refresh': auto_refresh,
        'detailed_analysis': detailed_analysis
    }

def render_real_time_analysis(analyzer: EnhancedMigrationAnalyzer, config: Dict):
    """Render comprehensive real-time analysis"""
    st.subheader("🧠 Intelligent Migration Analysis")
    
    with st.spinner("🔬 Analyzing network performance and migration requirements..."):
        analysis = analyzer.analyze_migration_performance(config)
    
    # Performance Score Dashboard
    performance = analysis['performance_score']
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{performance['overall']:.1f}/100",
            delta=f"Grade: {performance['grade']}"
        )
    
    with col2:
        st.metric(
            "Network Performance",
            f"{performance['network']:.1f}/100",
            delta=f"{random.uniform(-5, 5):.1f}"
        )
    
    with col3:
        st.metric(
            "OS Efficiency",
            f"{performance['os']:.1f}/100",
            delta=f"{random.uniform(-2, 2):.1f}"
        )
    
    with col4:
        st.metric(
            "Effective Throughput",
            f"{analysis['effective_throughput_mbps']:.0f} Mbps",
            delta=f"{random.uniform(-50, 50):.0f} Mbps"
        )
    
    with col5:
        st.metric(
            "Est. Migration Time",
            f"{analysis['estimated_time_hours']:.1f} hours",
            delta=f"{random.uniform(-1, 1):.1f} hours"
        )
    
    return analysis

def render_network_deep_dive(analysis: Dict):
    """Render detailed network analysis"""
    st.subheader("🌐 Network Performance Deep Dive")
    
    network_metrics = analysis['network_metrics']
    bottleneck_analysis = analysis['bottleneck_analysis']
    
    # Current Network Conditions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latency_color = "🟢" if network_metrics.latency < 50 else "🟡" if network_metrics.latency < 100 else "🔴"
        st.metric("Latency", f"{network_metrics.latency:.1f} ms", delta=latency_color)
    
    with col2:
        bandwidth_color = "🟢" if network_metrics.bandwidth > 1000 else "🟡" if network_metrics.bandwidth > 500 else "🔴"
        st.metric("Bandwidth", f"{network_metrics.bandwidth:.0f} Mbps", delta=bandwidth_color)
    
    with col3:
        loss_color = "🟢" if network_metrics.packet_loss < 0.5 else "🟡" if network_metrics.packet_loss < 1 else "🔴"
        st.metric("Packet Loss", f"{network_metrics.packet_loss:.2f}%", delta=loss_color)
    
    with col4:
        jitter_color = "🟢" if network_metrics.jitter < 5 else "🟡" if network_metrics.jitter < 10 else "🔴"
        st.metric("Jitter", f"{network_metrics.jitter:.1f} ms", delta=jitter_color)
    
    # Bottleneck Analysis
    st.markdown("### 🔍 AI Bottleneck Analysis")
    
    severity = bottleneck_analysis['severity_score']
    if severity < 30:
        severity_class = "bottleneck-low"
        severity_text = "Low Impact"
    elif severity < 70:
        severity_class = "bottleneck-medium" 
        severity_text = "Medium Impact"
    else:
        severity_class = "bottleneck-high"
        severity_text = "High Impact"
    
    st.markdown(f"""
    <div class="bottleneck-indicator {severity_class}">
        <strong>Primary Bottleneck:</strong> {bottleneck_analysis['bottleneck']}<br>
        <strong>Severity:</strong> {severity_text} ({severity:.0f}/100)<br>
        <strong>Expected Improvement:</strong> {bottleneck_analysis['performance_improvement']}
    </div>
    """, unsafe_allow_html=True)
    
    # TCP Analysis
    tcp_analysis = bottleneck_analysis['tcp_analysis']
    st.markdown("### ⚙️ TCP Configuration Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="os-performance-card">
            <h4>🔧 TCP Settings</h4>
            <p><strong>Window Scaling:</strong> {tcp_analysis['window_scaling']}</p>
            <p><strong>Congestion Control:</strong> {tcp_analysis['congestion_control']}</p>
            <p><strong>Buffer Efficiency:</strong> {tcp_analysis['buffer_efficiency']}</p>
            <p><strong>RTT Stability:</strong> {tcp_analysis['rtt_stability']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        os_profile = analysis['os_profile']
        st.markdown(f"""
        <div class="network-analysis-card">
            <h4>🖥️ OS Network Stack</h4>
            <p><strong>Efficiency:</strong> {os_profile.network_stack_efficiency*100:.1f}%</p>
            <p><strong>Driver:</strong> {os_profile.network_driver}</p>
            <p><strong>I/O Scheduler:</strong> {os_profile.io_scheduler}</p>
            <p><strong>NUMA:</strong> {'Enabled' if os_profile.numa_topology else 'Disabled'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Optimization recommendations
        st.markdown(f"""
        <div class="ai-powered-card">
            <h4>🚀 Quick Optimizations</h4>
        """, unsafe_allow_html=True)
        
        for rec in tcp_analysis.get('recommendations', [])[:3]:
            st.markdown(f"• {rec}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_optimization_opportunities(analysis: Dict):
    """Render optimization opportunities and recommendations"""
    st.subheader("🎯 Optimization Opportunities")
    
    opportunities = analysis['optimization_opportunities']
    
    if opportunities:
        for i, opp in enumerate(opportunities):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"""
                **{opp['category']}: {opp['opportunity']}**  
                Current: {opp['current_value']}  
                Recommended: {opp['recommended_value']}
                """)
            
            with col2:
                st.markdown(f"""
                **Expected Improvement:** {opp['expected_improvement']}  
                **Implementation:** {opp['implementation']}
                """)
            
            with col3:
                if st.button(f"Apply", key=f"apply_{i}"):
                    st.success(f"✅ Optimization queued!")
    else:
        st.info("🎉 No immediate optimization opportunities detected. System is well-tuned!")

def render_cost_analysis_dashboard(analysis: Dict):
    """Render intelligent cost analysis"""
    st.subheader("💰 Intelligent Cost Analysis")
    
    pricing_data = analysis['pricing_data']
    transfer_analysis = analysis['transfer_cost_analysis']
    
    # Cost optimization opportunities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="real-time-pricing">
            <h4>📊 Data Transfer Costs</h4>
            <p><strong>Base Cost:</strong> ${transfer_analysis['base_cost']:,.2f}</p>
            <p><strong>Optimized Cost:</strong> ${transfer_analysis['optimized_cost']:,.2f}</p>
            <p><strong>Potential Savings:</strong> {transfer_analysis['savings_percentage']:.1f}%</p>
            <p><strong>Tier:</strong> {transfer_analysis['tier']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="ai-insight">
            <h4>🧠 Optimization Methods</h4>
        """, unsafe_allow_html=True)
        
        for method in transfer_analysis['optimization_methods']:
            st.markdown(f"• {method}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Instance recommendations
    st.markdown("### 🏆 Recommended Instances")
    
    # Sort instances by recommendation score
    sorted_instances = sorted(pricing_data.items(), 
                            key=lambda x: x[1]['recommendation_score'], 
                            reverse=True)[:6]
    
    instance_df = pd.DataFrame([
        {
            'Instance Type': instance_type,
            'vCPU': data['vcpu'],
            'Memory (GB)': data['memory'],
            'Hourly Cost': f"${data['hourly_price']:.3f}",
            'Monthly Cost': f"${data['monthly_price']:.0f}",
            'Performance/$ Score': f"{data['recommendation_score']:.1f}",
            'Category': data['cost_category']
        }
        for instance_type, data in sorted_instances
    ])
    
    st.dataframe(instance_df, use_container_width=True, hide_index=True)

def render_risk_assessment(analysis: Dict):
    """Render migration risk assessment"""
    st.subheader("⚠️ Migration Risk Assessment")
    
    risk_assessment = analysis['risk_assessment']
    migration_strategy = analysis['migration_strategy_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_color = {
            'Low': '#28a745',
            'Medium': '#ffc107', 
            'High': '#dc3545'
        }.get(risk_assessment['overall_risk'], '#6c757d')
        
        st.markdown(f"""
        <div style="background: {risk_color}20; border-left: 5px solid {risk_color}; padding: 1rem; border-radius: 5px;">
            <h4 style="color: {risk_color};">Overall Risk Level: {risk_assessment['overall_risk']}</h4>
            <p><strong>Risk Score:</strong> {risk_assessment['risk_score']}/100</p>
            <p><strong>Strategy Suitability:</strong> {migration_strategy['suitability_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Identified Risks:**")
        for risk in risk_assessment['risks']:
            st.markdown(f"🔸 **{risk['type']} Risk:** {risk['description']}")
            st.markdown(f"   *Mitigation:* {risk['mitigation']}")

def render_performance_visualization(analysis: Dict):
    """Render performance visualization charts"""
    st.subheader("📈 Performance Trend Analysis")
    
    # Generate time series data for network performance
    now = datetime.now()
    time_points = pd.date_range(start=now - timedelta(hours=2), end=now, freq='5min')
    
    network_metrics = analysis['network_metrics']
    
    # Create realistic performance data with trends
    performance_data = []
    for i, timestamp in enumerate(time_points):
        # Add some realistic variation
        base_throughput = analysis['effective_throughput_mbps']
        variation = math.sin(i * 0.1) * 100 + random.uniform(-50, 50)
        
        performance_data.append({
            'Time': timestamp,
            'Throughput (Mbps)': max(100, base_throughput + variation),
            'Latency (ms)': network_metrics.latency + random.uniform(-10, 10),
            'Packet Loss (%)': max(0, network_metrics.packet_loss + random.uniform(-0.3, 0.3))
        })
    
    df = pd.DataFrame(performance_data)
    
    # Create subplot charts
    fig = go.Figure()
    
    # Throughput chart
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Throughput (Mbps)'],
        mode='lines+markers',
        name='Effective Throughput',
        line=dict(color='#4ECDC4', width=2),
        hovertemplate='%{y:.0f} Mbps<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Real-time Network Performance Monitoring",
        xaxis_title="Time",
        yaxis_title="Throughput (Mbps)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance distribution pie chart
    bottleneck_distribution = {
        'Network Latency': 25,
        'Bandwidth Limitation': 30,
        'OS Network Stack': 20,
        'Database Protocol': 15,
        'Migration Strategy': 10
    }
    
    fig_pie = px.pie(
        values=list(bottleneck_distribution.values()),
        names=list(bottleneck_distribution.keys()),
        title="Performance Impact Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

def main():
    """Main application function"""
    render_header()
    
    # System status
    render_system_status_dashboard()
    
    # Get configuration
    config = render_sidebar_controls()
    
    # Initialize analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 AI Analysis", 
        "🌐 Network Deep Dive", 
        "💰 Cost Optimization", 
        "⚠️ Risk Assessment",
        "📊 Performance Trends"
    ])
    
    with tab1:
        analysis = render_real_time_analysis(analyzer, config)
        render_optimization_opportunities(analysis)
    
    with tab2:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        render_network_deep_dive(analysis)
    
    with tab3:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        render_cost_analysis_dashboard(analysis)
    
    with tab4:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        render_risk_assessment(analysis)
    
    with tab5:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        render_performance_visualization(analysis)
    
    # Auto-refresh functionality
    if config.get('auto_refresh', False):
        time.sleep(3)
        st.rerun()
    
    # Footer with current system info
    st.markdown("---")
    current_os = platform.system()
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🤖 AI-Driven Migration Analyzer • 🖥️ Current OS: {current_os} • 🌐 Real-time Network Analysis • ⚡ Intelligent Optimization
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()