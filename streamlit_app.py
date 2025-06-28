import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
from typing import Dict, List, Tuple, Optional
import json
import requests
import os
import platform
import subprocess

# Page configuration
st.set_page_config(
    page_title="Flexible Enterprise Database Migration Analyzer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF9900 0%, #232F3E 50%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF9900;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .hardware-config-card {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(23,162,184,0.1);
    }
    
    .performance-impact-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(255,193,7,0.1);
    }
    
    .vmware-analysis-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(40,167,69,0.1);
    }
    
    .vrops-panel {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #6f42c1;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(111,66,193,0.1);
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 6px 20px rgba(102,126,234,0.2);
    }
    
    .network-factor-card {
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
    
    .api-status {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected { background-color: #28a745; animation: pulse 2s infinite; }
    .status-error { background-color: #dc3545; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .impact-indicator {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 2px 0;
        font-weight: bold;
    }
    
    .impact-high { background-color: #ffebee; color: #c62828; }
    .impact-medium { background-color: #fff3e0; color: #ef6c00; }
    .impact-low { background-color: #e8f5e8; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

class AIIntegration:
    """AI integration layer for intelligent analysis"""
    
    def __init__(self, anthropic_api_key: str = None):
        self.api_key = anthropic_api_key
        self.has_api = bool(anthropic_api_key)
    
    def analyze_hardware_impact(self, hardware_config: Dict, network_factors: Dict, performance_data: Dict) -> Dict:
        """AI-powered analysis of hardware configuration impact"""
        
        if self.has_api:
            return self._call_anthropic_api(hardware_config, network_factors, performance_data)
        else:
            return self._intelligent_hardware_analysis(hardware_config, network_factors, performance_data)
    
    def _intelligent_hardware_analysis(self, hardware_config: Dict, network_factors: Dict, performance_data: Dict) -> Dict:
        """Intelligent hardware impact analysis"""
        
        bottlenecks = []
        recommendations = []
        severity_score = 0
        
        # RAM Analysis
        ram_gb = hardware_config['ram_gb']
        if ram_gb < 16:
            bottlenecks.append(f"Low RAM ({ram_gb}GB) may cause memory pressure")
            recommendations.append("Consider increasing RAM to at least 32GB for optimal performance")
            severity_score += 25
        elif ram_gb < 32:
            bottlenecks.append(f"Moderate RAM ({ram_gb}GB) may limit concurrent operations")
            recommendations.append("Increase RAM to 64GB for better concurrent connection handling")
            severity_score += 15
        
        # CPU Analysis
        cpu_cores = hardware_config['cpu_cores']
        cpu_ghz = hardware_config['cpu_ghz']
        cpu_score = cpu_cores * cpu_ghz
        
        if cpu_score < 8:  # Less than 4 cores @ 2GHz equivalent
            bottlenecks.append(f"Low CPU capacity ({cpu_cores} cores @ {cpu_ghz}GHz)")
            recommendations.append("Increase CPU cores or clock speed for better processing")
            severity_score += 30
        
        # NIC Analysis
        nic_type = hardware_config['nic_type']
        nic_speed = hardware_config['nic_speed']
        
        if nic_speed < 1000:  # Less than 1Gbps
            bottlenecks.append(f"Network bottleneck: {nic_type} at {nic_speed}Mbps")
            recommendations.append("Upgrade to 10Gbps NIC for better migration performance")
            severity_score += 35
        elif nic_speed < 10000:  # Less than 10Gbps
            bottlenecks.append(f"Network constraint: {nic_type} at {nic_speed}Mbps")
            recommendations.append("Consider 25Gbps NIC for large database migrations")
            severity_score += 20
        
        # VMware vs Physical Analysis
        if hardware_config['server_type'] == 'vmware':
            vm_overhead = self._calculate_vmware_overhead(hardware_config)
            if vm_overhead > 0.25:  # More than 25% overhead
                bottlenecks.append(f"High VMware virtualization overhead ({vm_overhead*100:.1f}%)")
                recommendations.append("Consider physical deployment or optimize VM configuration")
                severity_score += 20
        
        # Network Factors Impact
        if network_factors['packet_loss'] > 0.5:
            bottlenecks.append(f"Network packet loss ({network_factors['packet_loss']:.2f}%)")
            recommendations.append("Investigate network infrastructure for packet loss")
            severity_score += 25
        
        if network_factors['latency'] > 50:
            bottlenecks.append(f"High network latency ({network_factors['latency']:.1f}ms)")
            recommendations.append("Optimize network routing or consider edge deployment")
            severity_score += 20
        
        # Performance improvement prediction
        improvement_potential = min(60, max(10, len(recommendations) * 8))
        
        return {
            'primary_bottleneck': bottlenecks[0] if bottlenecks else "No significant bottlenecks detected",
            'all_bottlenecks': bottlenecks,
            'recommendations': recommendations[:6],
            'severity_score': min(100, severity_score),
            'improvement_potential': f"{improvement_potential}-{improvement_potential + 15}%",
            'hardware_optimization_score': max(0, 100 - severity_score),
            'vmware_impact': self._calculate_vmware_overhead(hardware_config) if hardware_config['server_type'] == 'vmware' else 0
        }
    
    def _calculate_vmware_overhead(self, hardware_config: Dict) -> float:
        """Calculate VMware virtualization overhead"""
        base_overhead = 0.15  # 15% base overhead
        
        # RAM overhead (more RAM = better virtualization efficiency)
        ram_factor = min(0.1, max(0, (32 - hardware_config['ram_gb']) / 320))
        
        # CPU overhead (more cores = better efficiency)
        cpu_factor = min(0.1, max(0, (8 - hardware_config['cpu_cores']) / 80))
        
        # NIC overhead (faster NIC = less overhead)
        nic_factor = min(0.1, max(0, (10000 - hardware_config['nic_speed']) / 100000))
        
        total_overhead = base_overhead + ram_factor + cpu_factor + nic_factor
        return min(0.4, total_overhead)  # Cap at 40% overhead

class FlexibleHardwareManager:
    """Manage flexible hardware configurations"""
    
    def __init__(self):
        # NIC Types and their characteristics
        self.nic_types = {
            'gigabit_copper': {
                'name': 'Gigabit Ethernet (Copper)',
                'max_speed': 1000,
                'latency_factor': 1.0,
                'cpu_overhead': 0.08,
                'reliability': 0.98,
                'cost_factor': 1.0
            },
            'gigabit_fiber': {
                'name': 'Gigabit Ethernet (Fiber)',
                'max_speed': 1000,
                'latency_factor': 0.95,
                'cpu_overhead': 0.06,
                'reliability': 0.99,
                'cost_factor': 1.5
            },
            '10g_copper': {
                'name': '10Gb Ethernet (Copper)',
                'max_speed': 10000,
                'latency_factor': 0.8,
                'cpu_overhead': 0.12,
                'reliability': 0.985,
                'cost_factor': 3.0
            },
            '10g_fiber': {
                'name': '10Gb Ethernet (Fiber)',
                'max_speed': 10000,
                'latency_factor': 0.75,
                'cpu_overhead': 0.10,
                'reliability': 0.995,
                'cost_factor': 4.0
            },
            '25g_fiber': {
                'name': '25Gb Ethernet (Fiber)',
                'max_speed': 25000,
                'latency_factor': 0.7,
                'cpu_overhead': 0.15,
                'reliability': 0.997,
                'cost_factor': 8.0
            },
            '40g_fiber': {
                'name': '40Gb Ethernet (Fiber)',
                'max_speed': 40000,
                'latency_factor': 0.65,
                'cpu_overhead': 0.18,
                'reliability': 0.998,
                'cost_factor': 12.0
            }
        }
        
        # Server platform characteristics
        self.server_platforms = {
            'physical': {
                'name': 'Physical Server',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.95,
                'io_efficiency': 1.0,
                'network_efficiency': 0.98,
                'base_overhead': 0.02
            },
            'vmware': {
                'name': 'VMware Virtual Machine',
                'cpu_efficiency': 0.82,
                'memory_efficiency': 0.78,
                'io_efficiency': 0.75,
                'network_efficiency': 0.80,
                'base_overhead': 0.18
            }
        }
    
    def calculate_hardware_performance(self, config: Dict) -> Dict:
        """Calculate performance based on hardware configuration"""
        
        platform = self.server_platforms[config['server_type']]
        nic = self.nic_types[config['nic_type']]
        
        # RAM Performance Impact
        ram_gb = config['ram_gb']
        ram_performance = min(1.0, ram_gb / 64)  # Optimal at 64GB+
        
        # CPU Performance Impact
        cpu_cores = config['cpu_cores']
        cpu_ghz = config['cpu_ghz']
        cpu_performance = min(1.0, (cpu_cores * cpu_ghz) / 32)  # Optimal at 8 cores @ 4GHz
        
        # NIC Performance Impact
        nic_performance = min(1.0, config['nic_speed'] / nic['max_speed'])
        
        # Platform efficiency
        platform_efficiency = (
            platform['cpu_efficiency'] * platform['memory_efficiency'] * 
            platform['io_efficiency'] * platform['network_efficiency']
        )
        
        # Combined performance score
        overall_performance = (
            ram_performance * 0.25 +
            cpu_performance * 0.30 +
            nic_performance * 0.25 +
            platform_efficiency * 0.20
        )
        
        # Calculate actual throughput
        theoretical_throughput = config['nic_speed'] * nic_performance
        actual_throughput = theoretical_throughput * overall_performance * (1 - nic['cpu_overhead'])
        
        # VMware specific calculations
        vmware_impact = 0
        if config['server_type'] == 'vmware':
            vmware_impact = self._calculate_detailed_vmware_impact(config)
            actual_throughput *= (1 - vmware_impact)
        
        return {
            'ram_performance': ram_performance,
            'cpu_performance': cpu_performance,
            'nic_performance': nic_performance,
            'platform_efficiency': platform_efficiency,
            'overall_performance': overall_performance,
            'theoretical_throughput': theoretical_throughput,
            'actual_throughput': actual_throughput,
            'vmware_impact': vmware_impact,
            'nic_characteristics': nic,
            'platform_characteristics': platform
        }
    
    def _calculate_detailed_vmware_impact(self, config: Dict) -> float:
        """Calculate detailed VMware impact based on configuration"""
        
        base_impact = 0.15
        
        # Memory impact (less RAM = more overhead)
        if config['ram_gb'] < 32:
            base_impact += 0.08
        elif config['ram_gb'] < 16:
            base_impact += 0.15
        
        # CPU impact (fewer cores = more overhead per core)
        if config['cpu_cores'] < 4:
            base_impact += 0.10
        elif config['cpu_cores'] < 8:
            base_impact += 0.05
        
        # NIC impact (lower speed = more relative overhead)
        if config['nic_speed'] < 1000:
            base_impact += 0.12
        elif config['nic_speed'] < 10000:
            base_impact += 0.06
        
        return min(0.4, base_impact)  # Cap at 40%

class NetworkFactorsAnalyzer:
    """Analyze real-time networking factors"""
    
    def __init__(self):
        self.network_protocols = {
            'tcp': {'overhead': 0.05, 'reliability': 0.99, 'efficiency': 0.95},
            'udp': {'overhead': 0.02, 'reliability': 0.95, 'efficiency': 0.98},
            'rdma': {'overhead': 0.01, 'reliability': 0.999, 'efficiency': 0.99}
        }
    
    def get_real_time_network_factors(self, hardware_config: Dict) -> Dict:
        """Get real-time network factors affecting performance"""
        
        current_time = datetime.now()
        hour = current_time.hour
        
        # Business hours effect
        business_multiplier = 1.6 if 9 <= hour <= 17 else 0.7
        
        # Base network conditions with realistic variation
        base_latency = 15 + math.sin(time.time() / 1000) * 8
        base_congestion = 20 + math.sin(time.time() / 2000) * 15
        
        # Apply business hours effect
        latency = base_latency * (1 + (business_multiplier - 1) * 0.3) + random.uniform(-3, 8)
        congestion = base_congestion * business_multiplier + random.uniform(-5, 15)
        congestion = max(0, min(95, congestion))
        
        # Packet loss correlates with congestion
        packet_loss = max(0, (congestion / 100) * 1.5 + random.uniform(-0.2, 0.8))
        
        # Jitter varies with network load
        jitter = 1 + (congestion / 15) + random.uniform(0, 5)
        
        # Bandwidth utilization
        nic_speed = hardware_config['nic_speed']
        available_bandwidth = nic_speed * (1 - congestion / 100) * (1 - packet_loss / 100)
        
        # MTU impact based on NIC type
        mtu_size = 9000 if 'fiber' in hardware_config['nic_type'] and nic_speed >= 10000 else 1500
        
        # Protocol efficiency
        protocol_efficiency = self.network_protocols['tcp']['efficiency']
        
        # Calculate network quality score
        network_quality = (
            max(0, 100 - latency) * 0.3 +
            max(0, 100 - congestion) * 0.3 +
            max(0, 100 - packet_loss * 20) * 0.2 +
            max(0, 100 - jitter * 5) * 0.2
        )
        
        return {
            'latency': latency,
            'congestion': congestion,
            'packet_loss': packet_loss,
            'jitter': jitter,
            'available_bandwidth': available_bandwidth,
            'mtu_size': mtu_size,
            'protocol_efficiency': protocol_efficiency,
            'network_quality': network_quality,
            'business_hours_factor': business_multiplier
        }

class FlexibleMigrationAnalyzer:
    """Enhanced migration analyzer with flexible hardware configuration"""
    
    def __init__(self, anthropic_api_key: str = None):
        self.ai_integration = AIIntegration(anthropic_api_key)
        self.hardware_manager = FlexibleHardwareManager()
        self.network_analyzer = NetworkFactorsAnalyzer()
        
        # Database engines with network characteristics
        self.database_engines = {
            'mysql': {
                'name': 'MySQL',
                'memory_efficiency': 0.85,
                'cpu_efficiency': 0.88,
                'network_sensitivity': 0.7,
                'connection_overhead_kb': 4,
                'optimal_ram_per_gb_db': 0.1,
                'cpu_intensive': False,
                'supports_compression': True
            },
            'postgresql': {
                'name': 'PostgreSQL',
                'memory_efficiency': 0.90,
                'cpu_efficiency': 0.92,
                'network_sensitivity': 0.75,
                'connection_overhead_kb': 8,
                'optimal_ram_per_gb_db': 0.15,
                'cpu_intensive': True,
                'supports_compression': True
            },
            'oracle': {
                'name': 'Oracle Database',
                'memory_efficiency': 0.95,
                'cpu_efficiency': 0.94,
                'network_sensitivity': 0.6,
                'connection_overhead_kb': 12,
                'optimal_ram_per_gb_db': 0.2,
                'cpu_intensive': True,
                'supports_compression': True
            },
            'sqlserver': {
                'name': 'SQL Server',
                'memory_efficiency': 0.88,
                'cpu_efficiency': 0.90,
                'network_sensitivity': 0.65,
                'connection_overhead_kb': 6,
                'optimal_ram_per_gb_db': 0.12,
                'cpu_intensive': False,
                'supports_compression': True
            },
            'mongodb': {
                'name': 'MongoDB',
                'memory_efficiency': 0.82,
                'cpu_efficiency': 0.85,
                'network_sensitivity': 0.8,
                'connection_overhead_kb': 3,
                'optimal_ram_per_gb_db': 0.08,
                'cpu_intensive': False,
                'supports_compression': True
            }
        }
        
        # Migration strategies
        self.migration_strategies = {
            'online': {
                'name': 'Online Migration',
                'network_requirement': 'high',
                'cpu_requirement': 'medium',
                'memory_requirement': 'high',
                'downtime_minutes': 2,
                'complexity': 8
            },
            'offline': {
                'name': 'Offline Migration',
                'network_requirement': 'medium',
                'cpu_requirement': 'low',
                'memory_requirement': 'medium',
                'downtime_minutes': 120,
                'complexity': 4
            },
            'hybrid': {
                'name': 'Hybrid Migration',
                'network_requirement': 'medium',
                'cpu_requirement': 'medium',
                'memory_requirement': 'medium',
                'downtime_minutes': 20,
                'complexity': 6
            }
        }
    
    def analyze_migration_performance(self, config: Dict) -> Dict:
        """Comprehensive migration analysis with flexible hardware"""
        
        # Get hardware performance characteristics
        hardware_perf = self.hardware_manager.calculate_hardware_performance(config)
        
        # Get real-time network factors
        network_factors = self.network_analyzer.get_real_time_network_factors(config)
        
        # Database engine analysis
        db_engine = self.database_engines[config['database_engine']]
        migration_strategy = self.migration_strategies[config['migration_strategy']]
        
        # Calculate effective throughput
        base_throughput = hardware_perf['actual_throughput']
        
        # Apply database-specific factors
        db_network_penalty = 1 - (db_engine['network_sensitivity'] * network_factors['packet_loss'] / 100)
        db_efficiency = db_engine['memory_efficiency'] * db_engine['cpu_efficiency']
        
        # Network quality impact
        network_quality_factor = network_factors['network_quality'] / 100
        
        # Migration strategy impact
        strategy_efficiency = self._calculate_strategy_efficiency(migration_strategy, config, network_factors)
        
        # Final throughput calculation
        effective_throughput = (
            base_throughput * 
            db_network_penalty * 
            db_efficiency * 
            network_quality_factor * 
            strategy_efficiency
        )
        
        # Compression benefit
        if config.get('enable_compression', True) and db_engine['supports_compression']:
            effective_throughput *= 1.3
        
        # Calculate migration time
        database_size_gb = config['database_size_gb']
        estimated_time_hours = (database_size_gb * 8 * 1000) / (effective_throughput * 3600)
        
        # Resource utilization analysis
        resource_utilization = self._analyze_resource_utilization(config, db_engine)
        
        # AI-powered analysis
        ai_analysis = self.ai_integration.analyze_hardware_impact(
            config, network_factors, {'throughput': effective_throughput}
        )
        
        return {
            'hardware_performance': hardware_perf,
            'network_factors': network_factors,
            'effective_throughput': effective_throughput,
            'estimated_time_hours': estimated_time_hours,
            'resource_utilization': resource_utilization,
            'ai_analysis': ai_analysis,
            'db_efficiency': db_efficiency,
            'strategy_efficiency': strategy_efficiency,
            'vmware_vs_physical_impact': self._compare_vmware_vs_physical(config)
        }
    
    def _calculate_strategy_efficiency(self, strategy: Dict, config: Dict, network_factors: Dict) -> float:
        """Calculate migration strategy efficiency based on current conditions"""
        
        base_efficiency = 0.8
        
        # Network requirement matching
        if strategy['network_requirement'] == 'high' and network_factors['network_quality'] < 70:
            base_efficiency *= 0.7
        elif strategy['network_requirement'] == 'medium' and network_factors['network_quality'] < 50:
            base_efficiency *= 0.8
        
        # Hardware capability matching
        if strategy['cpu_requirement'] == 'high' and config['cpu_cores'] < 8:
            base_efficiency *= 0.8
        if strategy['memory_requirement'] == 'high' and config['ram_gb'] < 32:
            base_efficiency *= 0.8
        
        return base_efficiency
    
    def _analyze_resource_utilization(self, config: Dict, db_engine: Dict) -> Dict:
        """Analyze resource utilization patterns"""
        
        database_size_gb = config['database_size_gb']
        
        # RAM analysis
        optimal_ram = database_size_gb * db_engine['optimal_ram_per_gb_db']
        current_ram = config['ram_gb']
        ram_adequacy = min(1.0, current_ram / optimal_ram) if optimal_ram > 0 else 1.0
        
        # CPU analysis
        estimated_cpu_load = min(90, (database_size_gb / 1000) * 20 + (config.get('concurrent_connections', 100) / 10))
        cpu_adequacy = min(1.0, (config['cpu_cores'] * config['cpu_ghz']) / (estimated_cpu_load / 10))
        
        # Network analysis
        required_bandwidth = min(config['nic_speed'], database_size_gb * 0.5)  # Rough estimate
        bandwidth_adequacy = min(1.0, config['nic_speed'] / required_bandwidth) if required_bandwidth > 0 else 1.0
        
        return {
            'ram_adequacy': ram_adequacy,
            'cpu_adequacy': cpu_adequacy,
            'bandwidth_adequacy': bandwidth_adequacy,
            'optimal_ram_gb': optimal_ram,
            'estimated_cpu_load': estimated_cpu_load,
            'required_bandwidth_mbps': required_bandwidth
        }
    
    def _compare_vmware_vs_physical(self, config: Dict) -> Dict:
        """Compare VMware vs Physical performance for current configuration"""
        
        # Create physical equivalent config
        physical_config = config.copy()
        physical_config['server_type'] = 'physical'
        
        # Calculate both scenarios
        vmware_perf = self.hardware_manager.calculate_hardware_performance(config)
        physical_perf = self.hardware_manager.calculate_hardware_performance(physical_config)
        
        # Performance difference
        throughput_difference = physical_perf['actual_throughput'] - vmware_perf['actual_throughput']
        performance_gap = (throughput_difference / physical_perf['actual_throughput']) * 100
        
        return {
            'vmware_throughput': vmware_perf['actual_throughput'],
            'physical_throughput': physical_perf['actual_throughput'],
            'performance_gap_percent': performance_gap,
            'vmware_overhead': vmware_perf['vmware_impact'] * 100,
            'recommendation': 'Physical' if performance_gap > 20 else 'VMware acceptable'
        }

def render_header():
    """Render the flexible migration analyzer header"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Flexible Enterprise Database Migration Analyzer</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Configure RAM • CPU Cores • NIC Types • Physical vs VMware • Real-time Network Impact</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Dynamic Hardware Analysis • AI-Powered Recommendations • Real-time Performance Impact</p>
    </div>
    """, unsafe_allow_html=True)

def render_flexible_sidebar_controls():
    """Render flexible hardware configuration sidebar"""
    st.sidebar.header("⚙️ Hardware Configuration")
    
    # AI Configuration
    st.sidebar.subheader("🔑 AI Configuration")
    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key", 
        type="password",
        value=st.session_state.get('anthropic_api_key', ''),
        help="Enter your Anthropic API key for advanced AI analysis"
    )
    st.session_state['anthropic_api_key'] = anthropic_api_key
    
    # Server Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "Physical Server" if x == "physical" else "VMware Virtual Machine"
    )
    
    # RAM Configuration
    st.sidebar.subheader("💾 Memory Configuration")
    ram_gb = st.sidebar.selectbox(
        "RAM (GB)",
        [8, 16, 32, 64, 128, 256, 512, 1024],
        index=2,  # Default to 32GB
        help="Select total system RAM"
    )
    
    # CPU Configuration
    st.sidebar.subheader("🔧 CPU Configuration")
    cpu_cores = st.sidebar.selectbox(
        "CPU Cores",
        [2, 4, 8, 16, 24, 32, 48, 64],
        index=2,  # Default to 8 cores
        help="Number of CPU cores"
    )
    
    cpu_ghz = st.sidebar.selectbox(
        "CPU Clock Speed (GHz)",
        [2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4],
        index=3,  # Default to 3.2GHz
        help="CPU base clock speed"
    )
    
    # NIC Configuration
    st.sidebar.subheader("🌐 Network Interface")
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,  # Default to 10Gb Fiber
        format_func=lambda x: {
            'gigabit_copper': 'Gigabit Ethernet (Copper)',
            'gigabit_fiber': 'Gigabit Ethernet (Fiber)',
            '10g_copper': '10Gb Ethernet (Copper)',
            '10g_fiber': '10Gb Ethernet (Fiber)',
            '25g_fiber': '25Gb Ethernet (Fiber)',
            '40g_fiber': '40Gb Ethernet (Fiber)'
        }[x]
    )
    
    # Auto-populate NIC speed based on type
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000,
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    st.sidebar.info(f"NIC Speed: {nic_speed:,} Mbps")
    
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
        value=200
    )
    
    # Migration Strategy
    st.sidebar.subheader("🚀 Migration Strategy")
    migration_strategy = st.sidebar.selectbox(
        "Migration Type",
        ["online", "offline", "hybrid"],
        format_func=lambda x: {
            'online': 'Online (Minimal Downtime)',
            'offline': 'Offline (Maintenance Window)',
            'hybrid': 'Hybrid (Balanced)'
        }[x]
    )
    
    enable_compression = st.sidebar.checkbox("Enable Compression", value=True)
    
    # Real-time Controls
    st.sidebar.subheader("⚡ Real-time Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (3s)", value=True)
    show_comparison = st.sidebar.checkbox("Show VMware vs Physical", value=True)
    
    if st.sidebar.button("🔄 Refresh Analysis"):
        st.rerun()
    
    return {
        'anthropic_api_key': anthropic_api_key,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'concurrent_connections': concurrent_connections,
        'migration_strategy': migration_strategy,
        'enable_compression': enable_compression,
        'auto_refresh': auto_refresh,
        'show_comparison': show_comparison
    }

def render_hardware_configuration_display(config: Dict, analyzer: FlexibleMigrationAnalyzer):
    """Display current hardware configuration and its impact"""
    st.subheader("⚙️ Current Hardware Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="hardware-config-card">
            <h4>🖥️ Platform</h4>
            <p><strong>Type:</strong> {config['server_type'].title()}</p>
            <p><strong>CPU:</strong> {config['cpu_cores']} cores @ {config['cpu_ghz']}GHz</p>
            <p><strong>Total Power:</strong> {config['cpu_cores'] * config['cpu_ghz']:.1f} GHz</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="hardware-config-card">
            <h4>💾 Memory</h4>
            <p><strong>Total RAM:</strong> {config['ram_gb']} GB</p>
            <p><strong>Per Core:</strong> {config['ram_gb'] / config['cpu_cores']:.1f} GB</p>
            <p><strong>DB Ratio:</strong> {(config['ram_gb'] / config['database_size_gb']) * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        nic_name = analyzer.hardware_manager.nic_types[config['nic_type']]['name']
        st.markdown(f"""
        <div class="hardware-config-card">
            <h4>🌐 Network</h4>
            <p><strong>NIC:</strong> {nic_name}</p>
            <p><strong>Speed:</strong> {config['nic_speed']:,} Mbps</p>
            <p><strong>Bandwidth:</strong> {config['nic_speed'] / 1000:.1f} Gbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        db_name = analyzer.database_engines[config['database_engine']]['name']
        st.markdown(f"""
        <div class="hardware-config-card">
            <h4>💾 Database</h4>
            <p><strong>Engine:</strong> {db_name}</p>
            <p><strong>Size:</strong> {config['database_size_gb']:,} GB</p>
            <p><strong>Connections:</strong> {config['concurrent_connections']}</p>
        </div>
        """, unsafe_allow_html=True)

def render_real_time_performance_analysis(analyzer: FlexibleMigrationAnalyzer, config: Dict):
    """Render real-time performance analysis"""
    st.subheader("📊 Real-time Performance Analysis")
    
    with st.spinner("🔬 Analyzing hardware configuration and network conditions..."):
        analysis = analyzer.analyze_migration_performance(config)
    
    # Performance Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    hardware_perf = analysis['hardware_performance']
    network_factors = analysis['network_factors']
    
    with col1:
        st.metric(
            "Effective Throughput",
            f"{analysis['effective_throughput']:.0f} Mbps",
            delta=f"{random.uniform(-100, 100):.0f} Mbps"
        )
    
    with col2:
        st.metric(
            "Hardware Efficiency",
            f"{hardware_perf['overall_performance']*100:.1f}%",
            delta=f"{random.uniform(-2, 2):.1f}%"
        )
    
    with col3:
        st.metric(
            "Network Quality",
            f"{network_factors['network_quality']:.1f}/100",
            delta=f"{random.uniform(-5, 5):.1f}"
        )
    
    with col4:
        st.metric(
            "Migration Time",
            f"{analysis['estimated_time_hours']:.1f} hours",
            delta=f"{random.uniform(-1, 1):.1f} hours"
        )
    
    with col5:
        if config['server_type'] == 'vmware':
            st.metric(
                "VMware Overhead",
                f"{analysis['hardware_performance']['vmware_impact']*100:.1f}%",
                delta="vs Physical"
            )
        else:
            st.metric(
                "Platform Efficiency",
                f"{hardware_perf['platform_efficiency']*100:.1f}%",
                delta="Physical"
            )
    
    return analysis

def render_network_factors_analysis(network_factors: Dict):
    """Render real-time network factors analysis"""
    st.subheader("🌐 Real-time Network Factors")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latency_color = "🟢" if network_factors['latency'] < 30 else "🟡" if network_factors['latency'] < 60 else "🔴"
        st.metric("Latency", f"{network_factors['latency']:.1f} ms", delta=latency_color)
    
    with col2:
        congestion_color = "🟢" if network_factors['congestion'] < 30 else "🟡" if network_factors['congestion'] < 60 else "🔴"
        st.metric("Congestion", f"{network_factors['congestion']:.1f}%", delta=congestion_color)
    
    with col3:
        loss_color = "🟢" if network_factors['packet_loss'] < 0.5 else "🟡" if network_factors['packet_loss'] < 1 else "🔴"
        st.metric("Packet Loss", f"{network_factors['packet_loss']:.2f}%", delta=loss_color)
    
    with col4:
        st.metric("Available BW", f"{network_factors['available_bandwidth']:.0f} Mbps", f"MTU: {network_factors['mtu_size']}")
    
    # Network impact analysis
    st.markdown(f"""
    <div class="network-factor-card">
        <h4>📈 Network Impact Analysis</h4>
        <p><strong>Business Hours Factor:</strong> {network_factors['business_hours_factor']:.1f}x</p>
        <p><strong>Protocol Efficiency:</strong> {network_factors['protocol_efficiency']*100:.1f}%</p>
        <p><strong>Overall Network Quality:</strong> {network_factors['network_quality']:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)

def render_vmware_vs_physical_comparison(analysis: Dict, config: Dict):
    """Render VMware vs Physical comparison"""
    if not config.get('show_comparison', True):
        return
    
    st.subheader("⚖️ VMware vs Physical Performance Comparison")
    
    comparison = analysis['vmware_vs_physical_impact']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="vmware-analysis-card">
            <h4>🖥️ Physical Server</h4>
            <p><strong>Throughput:</strong> {comparison['physical_throughput']:.0f} Mbps</p>
            <p><strong>Efficiency:</strong> 98%</p>
            <p><strong>Overhead:</strong> 2%</p>
            <p style="color: #28a745;"><strong>Pros:</strong> Maximum performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="vmware-analysis-card">
            <h4>🔄 VMware VM</h4>
            <p><strong>Throughput:</strong> {comparison['vmware_throughput']:.0f} Mbps</p>
            <p><strong>Efficiency:</strong> {100 - comparison['vmware_overhead']:.1f}%</p>
            <p><strong>Overhead:</strong> {comparison['vmware_overhead']:.1f}%</p>
            <p style="color: #17a2b8;"><strong>Pros:</strong> Flexibility, HA</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        impact_class = "impact-high" if comparison['performance_gap_percent'] > 25 else "impact-medium" if comparison['performance_gap_percent'] > 15 else "impact-low"
        st.markdown(f"""
        <div class="performance-impact-card">
            <h4>📊 Performance Impact</h4>
            <p><strong>Performance Gap:</strong> {comparison['performance_gap_percent']:.1f}%</p>
            <div class="impact-indicator {impact_class}">
                {comparison['recommendation']}
            </div>
            <p><strong>Recommendation:</strong> {'Use Physical for max performance' if comparison['performance_gap_percent'] > 20 else 'VMware acceptable for this workload'}</p>
        </div>
        """, unsafe_allow_html=True)

def render_resource_utilization_analysis(analysis: Dict, config: Dict):
    """Render resource utilization analysis"""
    st.subheader("📈 Resource Utilization Analysis")
    
    resource_util = analysis['resource_utilization']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ram_adequacy = resource_util['ram_adequacy']
        ram_color = "#28a745" if ram_adequacy >= 0.8 else "#ffc107" if ram_adequacy >= 0.6 else "#dc3545"
        st.markdown(f"""
        <div style="background: {ram_color}20; border-left: 5px solid {ram_color}; padding: 1rem; border-radius: 5px;">
            <h4>💾 RAM Analysis</h4>
            <p><strong>Current:</strong> {config['ram_gb']} GB</p>
            <p><strong>Optimal:</strong> {resource_util['optimal_ram_gb']:.1f} GB</p>
            <p><strong>Adequacy:</strong> {ram_adequacy*100:.1f}%</p>
            <div style="background: #ddd; border-radius: 10px; overflow: hidden;">
                <div style="background: {ram_color}; width: {min(100, ram_adequacy*100):.1f}%; height: 20px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cpu_adequacy = resource_util['cpu_adequacy']
        cpu_color = "#28a745" if cpu_adequacy >= 0.8 else "#ffc107" if cpu_adequacy >= 0.6 else "#dc3545"
        st.markdown(f"""
        <div style="background: {cpu_color}20; border-left: 5px solid {cpu_color}; padding: 1rem; border-radius: 5px;">
            <h4>🔧 CPU Analysis</h4>
            <p><strong>Cores:</strong> {config['cpu_cores']} @ {config['cpu_ghz']}GHz</p>
            <p><strong>Est. Load:</strong> {resource_util['estimated_cpu_load']:.1f}%</p>
            <p><strong>Adequacy:</strong> {cpu_adequacy*100:.1f}%</p>
            <div style="background: #ddd; border-radius: 10px; overflow: hidden;">
                <div style="background: {cpu_color}; width: {min(100, cpu_adequacy*100):.1f}%; height: 20px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bw_adequacy = resource_util['bandwidth_adequacy']
        bw_color = "#28a745" if bw_adequacy >= 0.8 else "#ffc107" if bw_adequacy >= 0.6 else "#dc3545"
        st.markdown(f"""
        <div style="background: {bw_color}20; border-left: 5px solid {bw_color}; padding: 1rem; border-radius: 5px;">
            <h4>🌐 Network Analysis</h4>
            <p><strong>Available:</strong> {config['nic_speed']:,} Mbps</p>
            <p><strong>Required:</strong> {resource_util['required_bandwidth_mbps']:.0f} Mbps</p>
            <p><strong>Adequacy:</strong> {bw_adequacy*100:.1f}%</p>
            <div style="background: #ddd; border-radius: 10px; overflow: hidden;">
                <div style="background: {bw_color}; width: {min(100, bw_adequacy*100):.1f}%; height: 20px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_ai_recommendations(analysis: Dict):
    """Render AI-powered recommendations"""
    if 'ai_analysis' not in analysis:
        return
    
    st.subheader("🤖 AI-Powered Hardware Optimization")
    
    ai_analysis = analysis['ai_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity_color = "#dc3545" if ai_analysis['severity_score'] > 70 else "#ffc107" if ai_analysis['severity_score'] > 40 else "#28a745"
        st.markdown(f"""
        <div class="ai-insight">
            <h4>🎯 Bottleneck Analysis</h4>
            <p><strong>Primary Issue:</strong> {ai_analysis['primary_bottleneck']}</p>
            <p><strong>Severity Score:</strong> {ai_analysis['severity_score']}/100</p>
            <p><strong>Improvement Potential:</strong> {ai_analysis['improvement_potential']}</p>
            <p><strong>Hardware Score:</strong> {ai_analysis['hardware_optimization_score']}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🔧 Optimization Recommendations:**")
        for i, rec in enumerate(ai_analysis['recommendations'][:4]):
            st.markdown(f"{i+1}. {rec}")
        
        if ai_analysis.get('vmware_impact', 0) > 0:
            st.markdown(f"**⚡ VMware Impact:** {ai_analysis['vmware_impact']*100:.1f}% overhead")

def render_performance_charts(analysis: Dict, config: Dict):
    """Render performance visualization charts"""
    st.subheader("📊 Performance Impact Visualization")
    
    # Hardware component impact chart
    hardware_perf = analysis['hardware_performance']
    
    components = ['RAM', 'CPU', 'NIC', 'Platform']
    performance_scores = [
        hardware_perf['ram_performance'] * 100,
        hardware_perf['cpu_performance'] * 100,
        hardware_perf['nic_performance'] * 100,
        hardware_perf['platform_efficiency'] * 100
    ]
    
    fig_bar = go.Figure(data=[
        go.Bar(
            x=components,
            y=performance_scores,
            marker_color=['#FF9900', '#232F3E', '#4ECDC4', '#17a2b8'],
            text=[f"{score:.1f}%" for score in performance_scores],
            textposition='auto'
        )
    ])
    
    fig_bar.update_layout(
        title="Hardware Component Performance Scores",
        yaxis_title="Performance Score (%)",
        height=400
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Network factors impact over time (simulated)
    time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='5min')
    network_data = []
    
    for i, timestamp in enumerate(time_points):
        variation = math.sin(i * 0.2) * 20 + random.uniform(-10, 10)
        network_data.append({
            'Time': timestamp,
            'Network Quality': max(20, min(100, analysis['network_factors']['network_quality'] + variation)),
            'Effective Throughput': max(100, analysis['effective_throughput'] + variation * 10),
            'Latency': max(5, analysis['network_factors']['latency'] + random.uniform(-5, 5))
        })
    
    df_network = pd.DataFrame(network_data)
    
    fig_network = go.Figure()
    
    fig_network.add_trace(go.Scatter(
        x=df_network['Time'],
        y=df_network['Effective Throughput'],
        mode='lines',
        name='Throughput (Mbps)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig_network.update_layout(
        title="Real-time Throughput Performance",
        xaxis_title="Time",
        yaxis_title="Throughput (Mbps)",
        height=400
    )
    
    st.plotly_chart(fig_network, use_container_width=True)

def main():
    """Main application function with flexible configuration"""
    render_header()
    
    # Get flexible configuration
    config = render_flexible_sidebar_controls()
    
    # Initialize analyzer
    analyzer = FlexibleMigrationAnalyzer(config.get('anthropic_api_key'))
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Hardware Analysis",
        "🌐 Network Impact", 
        "⚖️ Platform Comparison",
        "📊 Resource Utilization",
        "📈 Performance Trends"
    ])
    
    with tab1:
        # Hardware configuration display
        render_hardware_configuration_display(config, analyzer)
        
        # Real-time performance analysis
        analysis = render_real_time_performance_analysis(analyzer, config)
        
        # AI recommendations
        render_ai_recommendations(analysis)
    
    with tab2:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        
        # Network factors analysis
        render_network_factors_analysis(analysis['network_factors'])
        
        # Show how different NICs would perform
        st.subheader("🔄 NIC Performance Comparison")
        nic_comparison_data = []
        for nic_name, nic_specs in analyzer.hardware_manager.nic_types.items():
            test_config = config.copy()
            test_config['nic_type'] = nic_name
            test_config['nic_speed'] = nic_specs['max_speed']
            test_perf = analyzer.hardware_manager.calculate_hardware_performance(test_config)
            
            nic_comparison_data.append({
                'NIC Type': nic_specs['name'],
                'Max Speed (Mbps)': nic_specs['max_speed'],
                'Actual Throughput (Mbps)': f"{test_perf['actual_throughput']:.0f}",
                'CPU Overhead': f"{nic_specs['cpu_overhead']*100:.1f}%",
                'Reliability': f"{nic_specs['reliability']*100:.1f}%",
                'Cost Factor': f"{nic_specs['cost_factor']:.1f}x"
            })
        
        nic_df = pd.DataFrame(nic_comparison_data)
        st.dataframe(nic_df, use_container_width=True, hide_index=True)
    
    with tab3:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        
        # VMware vs Physical comparison
        render_vmware_vs_physical_comparison(analysis, config)
        
        # Show impact of different RAM/CPU configurations
        st.subheader("💾 RAM/CPU Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RAM Impact:**")
            ram_options = [16, 32, 64, 128, 256]
            ram_impacts = []
            
            for ram in ram_options:
                test_config = config.copy()
                test_config['ram_gb'] = ram
                test_perf = analyzer.hardware_manager.calculate_hardware_performance(test_config)
                ram_impacts.append({
                    'RAM (GB)': ram,
                    'Performance': f"{test_perf['overall_performance']*100:.1f}%",
                    'Throughput (Mbps)': f"{test_perf['actual_throughput']:.0f}"
                })
            
            st.dataframe(pd.DataFrame(ram_impacts), hide_index=True)
        
        with col2:
            st.markdown("**CPU Impact:**")
            cpu_options = [4, 8, 16, 24, 32]
            cpu_impacts = []
            
            for cpu in cpu_options:
                test_config = config.copy()
                test_config['cpu_cores'] = cpu
                test_perf = analyzer.hardware_manager.calculate_hardware_performance(test_config)
                cpu_impacts.append({
                    'CPU Cores': cpu,
                    'Performance': f"{test_perf['overall_performance']*100:.1f}%",
                    'Throughput (Mbps)': f"{test_perf['actual_throughput']:.0f}"
                })
            
            st.dataframe(pd.DataFrame(cpu_impacts), hide_index=True)
    
    with tab4:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        
        # Resource utilization analysis
        render_resource_utilization_analysis(analysis, config)
    
    with tab5:
        if 'analysis' not in locals():
            analysis = analyzer.analyze_migration_performance(config)
        
        # Performance charts
        render_performance_charts(analysis, config)
    
    # Auto-refresh functionality
    if config.get('auto_refresh', False):
        time.sleep(3)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        ⚙️ Flexible Enterprise Migration Analyzer • 🖥️ Physical vs VMware • 💾 Dynamic RAM/CPU • 🌐 NIC Impact Analysis • 📊 Real-time Performance
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()