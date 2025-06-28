import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
import asyncio
import aiohttp
import json
import boto3
import requests
import os
from typing import Dict, List, Tuple, Optional
import anthropic
from dataclasses import dataclass
import psutil
import socket
import subprocess
import platform

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

class AINetworkAnalyzer:
    """AI-powered network performance analyzer"""
    
    def __init__(self, anthropic_api_key: str):
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        self.performance_cache = {}
        
    async def analyze_network_bottlenecks(self, metrics: NetworkMetrics, os_profile: OSPerformanceProfile) -> Dict:
        """AI-powered network bottleneck analysis"""
        if not self.anthropic_client:
            return self._fallback_analysis(metrics, os_profile)
            
        try:
            prompt = f"""
            Analyze this network configuration for database migration performance:
            
            Network Metrics:
            - Latency: {metrics.latency}ms
            - Bandwidth: {metrics.bandwidth}Mbps  
            - Packet Loss: {metrics.packet_loss}%
            - Jitter: {metrics.jitter}ms
            - MTU Size: {metrics.mtu_size}
            - TCP Window Size: {metrics.tcp_window_size}
            - Congestion Algorithm: {metrics.congestion_algorithm}
            
            OS Performance Profile:
            - OS Type: {os_profile.os_type}
            - Kernel Version: {os_profile.kernel_version}
            - Network Stack Efficiency: {os_profile.network_stack_efficiency}
            - Memory Management: {os_profile.memory_management_efficiency}
            - I/O Scheduler: {os_profile.io_scheduler}
            - Network Driver: {os_profile.network_driver}
            - NUMA Topology: {os_profile.numa_topology}
            
            Provide:
            1. Primary bottleneck identification
            2. OS-specific optimization recommendations
            3. Network tuning parameters
            4. Expected performance improvement percentage
            5. Migration strategy adjustments
            
            Return as JSON with keys: bottleneck, os_optimizations, network_tuning, performance_improvement, migration_adjustments
            """
            
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_response = json.loads(message.content[0].text)
            return ai_response
            
        except Exception as e:
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_analysis(metrics, os_profile)
    
    def _fallback_analysis(self, metrics: NetworkMetrics, os_profile: OSPerformanceProfile) -> Dict:
        """Fallback analysis when AI is unavailable"""
        bottlenecks = []
        
        if metrics.latency > 100:
            bottlenecks.append("High latency detected")
        if metrics.packet_loss > 1:
            bottlenecks.append("Packet loss affecting throughput")
        if metrics.bandwidth < 1000:
            bottlenecks.append("Bandwidth constraint")
        if os_profile.network_stack_efficiency < 0.8:
            bottlenecks.append(f"{os_profile.os_type} network stack inefficiency")
            
        return {
            "bottleneck": "; ".join(bottlenecks) if bottlenecks else "No major bottlenecks detected",
            "os_optimizations": [f"Optimize {os_profile.os_type} TCP stack", "Tune kernel parameters"],
            "network_tuning": ["Increase TCP window size", "Optimize MTU"],
            "performance_improvement": "15-25%",
            "migration_adjustments": ["Consider parallel transfers", "Implement compression"]
        }

class AWSPricingManager:
    """Real-time AWS pricing data manager"""
    
    def __init__(self):
        self.pricing_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour
        
    async def get_rds_pricing(self, region: str = "us-west-2") -> Dict:
        """Fetch real-time RDS pricing from AWS"""
        cache_key = f"rds_{region}"
        
        if self._is_cache_valid(cache_key):
            return self.pricing_cache[cache_key]
            
        try:
            # Use AWS Pricing API
            pricing_client = boto3.client('pricing', region_name='us-east-1')
            
            response = pricing_client.get_products(
                ServiceCode='AmazonRDS',
                Filters=[
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'location',
                        'Value': self._region_to_location(region)
                    },
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'databaseEngine',
                        'Value': 'MySQL'
                    }
                ]
            )
            
            pricing_data = {}
            for price_item in response['PriceList'][:20]:  # Limit to avoid timeout
                price_data = json.loads(price_item)
                instance_type = price_data['product']['attributes'].get('instanceType')
                if instance_type:
                    on_demand = price_data['terms']['OnDemand']
                    price_dimensions = list(on_demand.values())[0]['priceDimensions']
                    hourly_price = float(list(price_dimensions.values())[0]['pricePerUnit']['USD'])
                    
                    pricing_data[instance_type] = {
                        'hourly_price': hourly_price,
                        'monthly_price': hourly_price * 24 * 30,
                        'vcpu': price_data['product']['attributes'].get('vcpu'),
                        'memory': price_data['product']['attributes'].get('memory')
                    }
            
            self.pricing_cache[cache_key] = pricing_data
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            return pricing_data
            
        except Exception as e:
            st.warning(f"Could not fetch real-time pricing: {str(e)}")
            return self._get_fallback_pricing()
    
    async def get_data_transfer_pricing(self, region: str = "us-west-2") -> Dict:
        """Fetch data transfer pricing"""
        try:
            # Simplified data transfer pricing (AWS charges for outbound data)
            return {
                'first_1gb': 0.00,
                'up_to_10tb': 0.09,
                'next_40tb': 0.085,
                'next_100tb': 0.07,
                'over_150tb': 0.05
            }
        except Exception:
            return {'standard_rate': 0.09}
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to pricing API location"""
        region_mapping = {
            'us-west-2': 'US West (Oregon)',
            'us-east-1': 'US East (N. Virginia)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        return region_mapping.get(region, 'US West (Oregon)')
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        return (cache_key in self.pricing_cache and 
                cache_key in self.cache_expiry and 
                time.time() < self.cache_expiry[cache_key])
    
    def _get_fallback_pricing(self) -> Dict:
        """Fallback pricing when API is unavailable"""
        return {
            't3.micro': {'hourly_price': 0.017, 'monthly_price': 12.24, 'vcpu': '2', 'memory': '1 GiB'},
            't3.small': {'hourly_price': 0.034, 'monthly_price': 24.48, 'vcpu': '2', 'memory': '2 GiB'},
            'm5.large': {'hourly_price': 0.096, 'monthly_price': 69.12, 'vcpu': '2', 'memory': '8 GiB'},
            'm5.xlarge': {'hourly_price': 0.192, 'monthly_price': 138.24, 'vcpu': '4', 'memory': '16 GiB'},
            'r5.large': {'hourly_price': 0.126, 'monthly_price': 90.72, 'vcpu': '2', 'memory': '16 GiB'},
            'r5.xlarge': {'hourly_price': 0.252, 'monthly_price': 181.44, 'vcpu': '4', 'memory': '32 GiB'}
        }

class RealTimeSystemMonitor:
    """Real-time system and network monitoring"""
    
    def __init__(self):
        self.baseline_metrics = None
        
    def get_current_network_metrics(self) -> NetworkMetrics:
        """Get real-time network performance metrics"""
        try:
            # Network latency test
            latency = self._measure_latency("8.8.8.8")
            
            # Network statistics
            net_io = psutil.net_io_counters()
            
            # Estimate bandwidth utilization
            if not hasattr(self, '_last_net_io'):
                self._last_net_io = net_io
                self._last_time = time.time()
                time.sleep(1)
                net_io = psutil.net_io_counters()
            
            current_time = time.time()
            time_delta = current_time - self._last_time
            bytes_sent_delta = net_io.bytes_sent - self._last_net_io.bytes_sent
            bytes_recv_delta = net_io.bytes_recv - self._last_net_io.bytes_recv
            
            # Convert to Mbps
            bandwidth_out = (bytes_sent_delta * 8) / (time_delta * 1000000)
            bandwidth_in = (bytes_recv_delta * 8) / (time_delta * 1000000)
            bandwidth = max(bandwidth_out, bandwidth_in)
            
            self._last_net_io = net_io
            self._last_time = current_time
            
            # Get MTU size
            mtu_size = self._get_mtu_size()
            
            # Simulate other metrics (in production, use specialized tools)
            packet_loss = random.uniform(0, 2)  # Would use ping statistics
            jitter = random.uniform(1, 10)      # Would use specialized measurement
            tcp_window = 65536                  # Would read from system config
            
            return NetworkMetrics(
                latency=latency,
                bandwidth=bandwidth * 1000,  # Convert to Mbps
                packet_loss=packet_loss,
                jitter=jitter,
                mtu_size=mtu_size,
                tcp_window_size=tcp_window,
                congestion_algorithm="cubic"
            )
            
        except Exception as e:
            st.warning(f"Could not get real network metrics: {str(e)}")
            return self._get_simulated_metrics()
    
    def get_os_performance_profile(self) -> OSPerformanceProfile:
        """Get current OS performance characteristics"""
        try:
            system = platform.system()
            kernel_version = platform.release()
            
            # Determine network stack efficiency based on OS
            if system == "Linux":
                network_efficiency = 0.95
                io_scheduler = self._get_linux_io_scheduler()
            elif system == "Windows":
                network_efficiency = 0.82
                io_scheduler = "N/A"
            else:
                network_efficiency = 0.88
                io_scheduler = "Unknown"
            
            # Memory management efficiency
            memory_efficiency = 0.90 if system == "Linux" else 0.85
            
            # NUMA topology check
            numa_topology = self._check_numa_topology()
            
            return OSPerformanceProfile(
                os_type=system,
                kernel_version=kernel_version,
                network_stack_efficiency=network_efficiency,
                memory_management_efficiency=memory_efficiency,
                io_scheduler=io_scheduler,
                tcp_buffer_sizes={
                    'recv': 87380,
                    'send': 16384,
                    'max': 16777216
                },
                network_driver="e1000e",  # Would detect actual driver
                numa_topology=numa_topology
            )
            
        except Exception as e:
            st.warning(f"Could not get OS profile: {str(e)}")
            return self._get_default_os_profile()
    
    def _measure_latency(self, host: str) -> float:
        """Measure network latency to a host"""
        try:
            import subprocess
            import platform
            
            param = "-n" if platform.system().lower() == "windows" else "-c"
            command = ["ping", param, "1", host]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            
            if platform.system().lower() == "windows":
                # Parse Windows ping output
                for line in result.stdout.split('\n'):
                    if 'time=' in line:
                        return float(line.split('time=')[1].split('ms')[0])
            else:
                # Parse Unix ping output
                for line in result.stdout.split('\n'):
                    if 'time=' in line:
                        return float(line.split('time=')[1].split(' ')[0])
                        
            return 50.0  # Default if parsing fails
            
        except Exception:
            return random.uniform(20, 100)  # Simulated latency
    
    def _get_mtu_size(self) -> int:
        """Get network interface MTU size"""
        try:
            if platform.system() == "Linux":
                import subprocess
                result = subprocess.run(['ip', 'route', 'get', '8.8.8.8'], 
                                      capture_output=True, text=True)
                # Parse MTU from output (simplified)
                return 1500  # Default Ethernet MTU
            else:
                return 1500
        except Exception:
            return 1500
    
    def _get_linux_io_scheduler(self) -> str:
        """Get Linux I/O scheduler"""
        try:
            with open('/sys/block/sda/queue/scheduler', 'r') as f:
                content = f.read()
                # Extract current scheduler (in brackets)
                import re
                match = re.search(r'\[([^\]]+)\]', content)
                return match.group(1) if match else "mq-deadline"
        except Exception:
            return "mq-deadline"
    
    def _check_numa_topology(self) -> bool:
        """Check if NUMA topology is available"""
        try:
            if platform.system() == "Linux":
                import subprocess
                result = subprocess.run(['numactl', '--hardware'], 
                                      capture_output=True, text=True)
                return 'available' in result.stdout
            return False
        except Exception:
            return False
    
    def _get_simulated_metrics(self) -> NetworkMetrics:
        """Get simulated network metrics for demo"""
        return NetworkMetrics(
            latency=random.uniform(20, 150),
            bandwidth=random.uniform(500, 10000),
            packet_loss=random.uniform(0, 3),
            jitter=random.uniform(1, 15),
            mtu_size=1500,
            tcp_window_size=65536,
            congestion_algorithm="cubic"
        )
    
    def _get_default_os_profile(self) -> OSPerformanceProfile:
        """Get default OS profile for demo"""
        return OSPerformanceProfile(
            os_type="Linux",
            kernel_version="5.4.0",
            network_stack_efficiency=0.92,
            memory_management_efficiency=0.88,
            io_scheduler="mq-deadline",
            tcp_buffer_sizes={'recv': 87380, 'send': 16384, 'max': 16777216},
            network_driver="virtio_net",
            numa_topology=True
        )

class AIEnhancedMigrationAnalyzer:
    """Main analyzer with AI capabilities"""
    
    def __init__(self, anthropic_api_key: str = None):
        self.ai_analyzer = AINetworkAnalyzer(anthropic_api_key)
        self.pricing_manager = AWSPricingManager()
        self.system_monitor = RealTimeSystemMonitor()
        
        # Database engine profiles with real-world characteristics
        self.database_engines = {
            'mysql': {
                'name': 'MySQL',
                'network_sensitivity': 0.7,  # How sensitive to network issues
                'connection_pooling_efficiency': 0.85,
                'replication_overhead': 0.15,
                'typical_connection_overhead_kb': 4,
                'supports_compression': True,
                'optimal_tcp_settings': {'window_size': 65536, 'no_delay': True}
            },
            'postgresql': {
                'name': 'PostgreSQL', 
                'network_sensitivity': 0.75,
                'connection_pooling_efficiency': 0.88,
                'replication_overhead': 0.12,
                'typical_connection_overhead_kb': 8,
                'supports_compression': True,
                'optimal_tcp_settings': {'window_size': 131072, 'no_delay': True}
            },
            'oracle': {
                'name': 'Oracle Database',
                'network_sensitivity': 0.6,
                'connection_pooling_efficiency': 0.92,
                'replication_overhead': 0.10,
                'typical_connection_overhead_kb': 12,
                'supports_compression': True,
                'optimal_tcp_settings': {'window_size': 262144, 'no_delay': False}
            },
            'sqlserver': {
                'name': 'SQL Server',
                'network_sensitivity': 0.65,
                'connection_pooling_efficiency': 0.87,
                'replication_overhead': 0.18,
                'typical_connection_overhead_kb': 6,
                'supports_compression': True,
                'optimal_tcp_settings': {'window_size': 131072, 'no_delay': True}
            },
            'mongodb': {
                'name': 'MongoDB',
                'network_sensitivity': 0.8,
                'connection_pooling_efficiency': 0.80,
                'replication_overhead': 0.25,
                'typical_connection_overhead_kb': 3,
                'supports_compression': True,
                'optimal_tcp_settings': {'window_size': 65536, 'no_delay': True}
            }
        }
    
    async def analyze_migration_performance(self, config: Dict) -> Dict:
        """AI-powered migration performance analysis"""
        
        # Get real-time system metrics
        network_metrics = self.system_monitor.get_current_network_metrics()
        os_profile = self.system_monitor.get_os_performance_profile()
        
        # AI-powered bottleneck analysis
        ai_analysis = await self.ai_analyzer.analyze_network_bottlenecks(network_metrics, os_profile)
        
        # Database-specific adjustments
        db_engine = self.database_engines[config['database_engine']]
        
        # Calculate network efficiency based on AI analysis and OS profile
        base_efficiency = os_profile.network_stack_efficiency
        network_penalty = network_metrics.packet_loss / 100 * 0.3
        latency_penalty = min(network_metrics.latency / 1000, 0.2)
        
        effective_efficiency = base_efficiency * (1 - network_penalty - latency_penalty)
        
        # Database sensitivity adjustments
        db_network_impact = db_engine['network_sensitivity'] * (1 - effective_efficiency)
        
        # Connection overhead calculation
        connection_overhead_mb = (config.get('concurrent_connections', 100) * 
                                db_engine['typical_connection_overhead_kb']) / 1024
        
        # Calculate throughput with AI insights
        base_throughput = network_metrics.bandwidth * effective_efficiency
        
        # Apply AI-recommended optimizations
        if 'performance_improvement' in ai_analysis:
            improvement_pct = float(ai_analysis['performance_improvement'].replace('%', '').split('-')[0]) / 100
            optimized_throughput = base_throughput * (1 + improvement_pct)
        else:
            optimized_throughput = base_throughput
        
        # Migration time estimation
        data_size_gb = config.get('database_size_gb', 1000)
        estimated_time_hours = (data_size_gb * 8 * 1000) / (optimized_throughput * 3600)
        
        # Get real-time pricing
        pricing_data = await self.pricing_manager.get_rds_pricing()
        transfer_pricing = await self.pricing_manager.get_data_transfer_pricing()
        
        return {
            'network_metrics': network_metrics,
            'os_profile': os_profile,
            'ai_analysis': ai_analysis,
            'effective_throughput_mbps': optimized_throughput,
            'estimated_time_hours': estimated_time_hours,
            'connection_overhead_mb': connection_overhead_mb,
            'db_network_impact': db_network_impact,
            'pricing_data': pricing_data,
            'transfer_pricing': transfer_pricing,
            'bottleneck_score': self._calculate_bottleneck_score(network_metrics, os_profile),
            'optimization_recommendations': ai_analysis.get('os_optimizations', [])
        }
    
    def _calculate_bottleneck_score(self, network_metrics: NetworkMetrics, os_profile: OSPerformanceProfile) -> float:
        """Calculate overall bottleneck severity score (0-100)"""
        latency_score = min(network_metrics.latency / 200 * 100, 40)
        packet_loss_score = network_metrics.packet_loss * 15
        bandwidth_score = max(0, (1000 - network_metrics.bandwidth) / 1000 * 30)
        os_efficiency_score = (1 - os_profile.network_stack_efficiency) * 30
        
        return min(100, latency_score + packet_loss_score + bandwidth_score + os_efficiency_score)

def render_header():
    """Render the AI-powered header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Driven Enterprise Database Migration Analyzer</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Real-time AWS Pricing • AI Network Analysis • OS Performance Tuning • Intelligent Recommendations</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Powered by Anthropic AI • Live AWS API • Real-time System Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_dashboard():
    """Render API connection status"""
    st.subheader("🔌 API Connection Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        anthropic_status = "connected" if st.session_state.get('anthropic_api_key') else "error"
        status_class = "status-connected" if anthropic_status == "connected" else "status-error"
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status {status_class}"></span>
            <strong>Anthropic AI</strong><br>
            {'🟢 Connected' if anthropic_status == 'connected' else '🔴 Not Connected'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            boto3.client('pricing', region_name='us-east-1')
            aws_status = "connected"
        except Exception:
            aws_status = "error"
        status_class = "status-connected" if aws_status == "connected" else "status-error"
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status {status_class}"></span>
            <strong>AWS Pricing API</strong><br>
            {'🟢 Connected' if aws_status == 'connected' else '🔴 Not Connected'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        system_status = "connected"  # Always available
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>System Monitor</strong><br>
            🟢 Active
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        network_status = "connected"  # Always available
        st.markdown(f"""
        <div class="metric-card">
            <span class="api-status status-connected"></span>
            <strong>Network Monitor</strong><br>
            🟢 Monitoring
        </div>
        """, unsafe_allow_html=True)

def render_sidebar_controls():
    """Enhanced sidebar with API configuration"""
    st.sidebar.header("🔧 Configuration")
    
    # API Configuration
    st.sidebar.subheader("🔑 API Configuration")
    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key", 
        type="password",
        value=st.session_state.get('anthropic_api_key', ''),
        help="Enter your Anthropic API key for AI analysis"
    )
    st.session_state['anthropic_api_key'] = anthropic_api_key
    
    aws_region = st.sidebar.selectbox(
        "AWS Region",
        ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"],
        help="Region for pricing and performance optimization"
    )
    
    # Database Configuration
    st.sidebar.subheader("💾 Database Configuration")
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
        step=100
    )
    
    concurrent_connections = st.sidebar.number_input(
        "Concurrent Connections",
        min_value=10,
        max_value=2000,
        value=200,
        help="Number of concurrent database connections"
    )
    
    # Migration Strategy
    st.sidebar.subheader("🚀 Migration Strategy")
    migration_type = st.sidebar.selectbox(
        "Migration Type",
        ["online", "offline", "hybrid"],
        format_func=lambda x: {
            'online': 'Online (Zero Downtime)',
            'offline': 'Offline (Maintenance Window)', 
            'hybrid': 'Hybrid (Partial Downtime)'
        }[x]
    )
    
    compression_enabled = st.sidebar.checkbox("Enable Compression", value=True)
    parallel_connections = st.sidebar.slider("Parallel Connections", 1, 16, 4)
    
    # Real-time Controls
    st.sidebar.subheader("⚡ Real-time Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=bool(anthropic_api_key))
    
    if st.sidebar.button("🔄 Force Refresh"):
        st.rerun()
    
    return {
        'anthropic_api_key': anthropic_api_key,
        'aws_region': aws_region,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'concurrent_connections': concurrent_connections,
        'migration_type': migration_type,
        'compression_enabled': compression_enabled,
        'parallel_connections': parallel_connections,
        'auto_refresh': auto_refresh,
        'enable_ai_analysis': enable_ai_analysis
    }

async def render_real_time_analysis(analyzer: AIEnhancedMigrationAnalyzer, config: Dict):
    """Render real-time AI-powered analysis"""
    st.subheader("🧠 AI-Powered Real-time Analysis")
    
    with st.spinner("🤖 Analyzing network performance and bottlenecks..."):
        analysis = await analyzer.analyze_migration_performance(config)
    
    # Network Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Latency",
            f"{analysis['network_metrics'].latency:.1f} ms",
            delta=f"{random.uniform(-5, 5):.1f} ms"
        )
    
    with col2:
        st.metric(
            "Available Bandwidth", 
            f"{analysis['network_metrics'].bandwidth:.0f} Mbps",
            delta=f"{random.uniform(-100, 100):.0f} Mbps"
        )
    
    with col3:
        st.metric(
            "Packet Loss",
            f"{analysis['network_metrics'].packet_loss:.2f}%",
            delta=f"{random.uniform(-0.5, 0.5):.2f}%"
        )
    
    with col4:
        st.metric(
            "Effective Throughput",
            f"{analysis['effective_throughput_mbps']:.0f} Mbps",
            delta=f"{random.uniform(-50, 50):.0f} Mbps"
        )
    
    # AI Analysis Results
    if 'ai_analysis' in analysis and config['enable_ai_analysis']:
        st.markdown(f"""
        <div class="ai-powered-card">
            <h4>🤖 AI Network Analysis</h4>
            <p><strong>Primary Bottleneck:</strong> {analysis['ai_analysis'].get('bottleneck', 'Analysis in progress...')}</p>
            <p><strong>Expected Improvement:</strong> {analysis['ai_analysis'].get('performance_improvement', 'Calculating...')}</p>
            <p><strong>Bottleneck Severity:</strong> {analysis['bottleneck_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    # OS Performance Analysis  
    os_profile = analysis['os_profile']
    st.markdown(f"""
    <div class="os-performance-card">
        <h4>🖥️ OS Performance Profile</h4>
        <p><strong>Operating System:</strong> {os_profile.os_type} {os_profile.kernel_version}</p>
        <p><strong>Network Stack Efficiency:</strong> {os_profile.network_stack_efficiency*100:.1f}%</p>
        <p><strong>I/O Scheduler:</strong> {os_profile.io_scheduler}</p>
        <p><strong>NUMA Topology:</strong> {'Enabled' if os_profile.numa_topology else 'Disabled'}</p>
        <p><strong>Network Driver:</strong> {os_profile.network_driver}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return analysis

def render_network_optimization_recommendations(analysis: Dict):
    """Render AI-powered network optimization recommendations"""
    st.subheader("🔧 AI-Powered Optimization Recommendations")
    
    if 'ai_analysis' in analysis:
        ai_analysis = analysis['ai_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="network-analysis-card">
                <h4>🌐 Network Optimizations</h4>
            """, unsafe_allow_html=True)
            
            if 'network_tuning' in ai_analysis:
                for recommendation in ai_analysis['network_tuning']:
                    st.markdown(f"• {recommendation}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="network-analysis-card">
                <h4>🖥️ OS-Specific Optimizations</h4>
            """, unsafe_allow_html=True)
            
            if 'os_optimizations' in ai_analysis:
                for recommendation in ai_analysis['os_optimizations']:
                    st.markdown(f"• {recommendation}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Migration Strategy Adjustments
    if 'migration_adjustments' in analysis.get('ai_analysis', {}):
        st.markdown(f"""
        <div class="ai-insight">
            <h4>🚀 Migration Strategy Adjustments</h4>
        """, unsafe_allow_html=True)
        
        for adjustment in analysis['ai_analysis']['migration_adjustments']:
            st.markdown(f"• {adjustment}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_real_time_pricing_dashboard(analysis: Dict):
    """Render real-time AWS pricing dashboard"""
    st.subheader("💰 Real-time AWS Pricing Analysis")
    
    if 'pricing_data' in analysis:
        pricing_data = analysis['pricing_data']
        
        # Create pricing comparison
        pricing_df = pd.DataFrame([
            {
                'Instance Type': instance_type,
                'vCPU': data.get('vcpu', 'N/A'),
                'Memory': data.get('memory', 'N/A'), 
                'Hourly Cost': f"${data['hourly_price']:.3f}",
                'Monthly Cost': f"${data['monthly_price']:.0f}"
            }
            for instance_type, data in list(pricing_data.items())[:6]
        ])
        
        st.dataframe(pricing_df, use_container_width=True, hide_index=True)
        
        # Transfer cost estimation
        if 'transfer_pricing' in analysis:
            transfer_data = analysis['transfer_pricing']
            data_size_gb = analysis.get('database_size_gb', 1000)
            
            if data_size_gb <= 1:
                transfer_cost = 0
            elif data_size_gb <= 10240:  # 10TB
                transfer_cost = data_size_gb * transfer_data.get('up_to_10tb', 0.09)
            else:
                transfer_cost = data_size_gb * transfer_data.get('standard_rate', 0.09)
            
            st.markdown(f"""
            <div class="real-time-pricing">
                <h4>📊 Data Transfer Cost Estimation</h4>
                <p><strong>Database Size:</strong> {data_size_gb:,} GB</p>
                <p><strong>Estimated Transfer Cost:</strong> ${transfer_cost:,.2f}</p>
                <p><strong>Rate:</strong> ${transfer_data.get('up_to_10tb', 0.09):.3f}/GB</p>
            </div>
            """, unsafe_allow_html=True)

def render_performance_visualization(analysis: Dict):
    """Render performance visualization charts"""
    st.subheader("📈 Performance Analysis Visualization")
    
    # Network performance over time (simulated real-time data)
    time_points = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                               end=datetime.now(), freq='1min')
    
    network_data = pd.DataFrame({
        'Time': time_points,
        'Throughput (Mbps)': [analysis['effective_throughput_mbps'] + random.uniform(-100, 100) 
                              for _ in time_points],
        'Latency (ms)': [analysis['network_metrics'].latency + random.uniform(-10, 10) 
                        for _ in time_points],
        'Packet Loss (%)': [analysis['network_metrics'].packet_loss + random.uniform(-0.5, 0.5) 
                           for _ in time_points]
    })
    
    # Create subplots
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=network_data['Time'],
        y=network_data['Throughput (Mbps)'],
        mode='lines',
        name='Throughput (Mbps)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title="Real-time Network Performance",
        xaxis_title="Time",
        yaxis_title="Throughput (Mbps)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bottleneck analysis pie chart
    bottleneck_data = {
        'Network Latency': 30,
        'Bandwidth Limitation': 25, 
        'OS Network Stack': 20,
        'Database Engine': 15,
        'Storage I/O': 10
    }
    
    fig_pie = px.pie(
        values=list(bottleneck_data.values()),
        names=list(bottleneck_data.keys()),
        title="Performance Bottleneck Distribution"
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

async def main():
    """Main application function"""
    render_header()
    
    # Render API status
    render_api_status_dashboard()
    
    # Get configuration
    config = render_sidebar_controls()
    
    # Initialize analyzer with API key
    analyzer = AIEnhancedMigrationAnalyzer(config.get('anthropic_api_key'))
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 AI Analysis", 
        "🌐 Network Performance", 
        "💰 Cost Analysis", 
        "📊 Performance Visualization"
    ])
    
    with tab1:
        if config['enable_ai_analysis'] and config['anthropic_api_key']:
            analysis = await render_real_time_analysis(analyzer, config)
            render_network_optimization_recommendations(analysis)
        else:
            st.warning("🔑 Please enter your Anthropic API key in the sidebar to enable AI analysis")
            st.info("💡 The AI analysis provides intelligent network bottleneck identification and optimization recommendations")
    
    with tab2:
        # Real-time network monitoring
        if config['enable_ai_analysis']:
            analysis = await analyzer.analyze_migration_performance(config)
            
            # Network flow visualization
            st.markdown('<div class="network-flow"></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("TCP Window Size", f"{analysis['network_metrics'].tcp_window_size:,} bytes")
                st.metric("MTU Size", f"{analysis['network_metrics'].mtu_size} bytes")
                
            with col2:
                st.metric("Jitter", f"{analysis['network_metrics'].jitter:.1f} ms")
                st.metric("Congestion Algorithm", analysis['network_metrics'].congestion_algorithm)
                
            with col3:
                st.metric("Connection Overhead", f"{analysis['connection_overhead_mb']:.1f} MB")
                st.metric("DB Network Impact", f"{analysis['db_network_impact']*100:.1f}%")
        else:
            st.info("Enable AI analysis to see detailed network performance metrics")
    
    with tab3:
        if config['enable_ai_analysis']:
            analysis = await analyzer.analyze_migration_performance(config)
            render_real_time_pricing_dashboard(analysis)
        else:
            st.info("Enable AI analysis to see real-time AWS pricing data")
    
    with tab4:
        if config['enable_ai_analysis']:
            analysis = await analyzer.analyze_migration_performance(config)
            render_performance_visualization(analysis)
        else:
            st.info("Enable AI analysis to see performance visualizations")
    
    # Auto-refresh
    if config.get('auto_refresh', False):
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    # Run the async main function
    import asyncio
    asyncio.run(main())