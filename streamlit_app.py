import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="VM vs Physical Migration Performance Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF9900 0%, #232F3E 100%);
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
    
    .vrops-panel {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #6f42c1;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(111,66,193,0.1);
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        font-style: italic;
        box-shadow: 0 2px 10px rgba(0,123,255,0.1);
    }
    
    .network-status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-good { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-danger { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class MigrationPerformanceCalculator:
    """Enhanced calculator for VM vs Physical server migration performance"""
    
    def __init__(self):
        # Server performance characteristics
        self.server_configs = {
            'physical': {
                'name': 'Physical Server',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_throughput': 1.0,
                'network_overhead': 0.05,
                'reliability': 0.98
            },
            'virtual': {
                'name': 'Virtual Machine',
                'cpu_efficiency': 0.78,
                'memory_efficiency': 0.75,
                'io_throughput': 0.70,
                'network_overhead': 0.18,
                'reliability': 0.95
            }
        }
        
        # Storage configurations
        self.storage_configs = {
            'nas-linux': {
                'name': 'NAS Linux (Red Hat)',
                'os': 'linux',
                'memory': 48,
                'ethernet_adapter': 'VMWARE',
                'base_efficiency': 0.92,
                'io_throughput': 1.0,
                'cpu_overhead': 0.08
            },
            'windows-share': {
                'name': 'Windows Share',
                'os': 'windows',
                'memory': 80,
                'ethernet_adapter': 'VMAX',
                'base_efficiency': 0.78,
                'io_throughput': 0.85,
                'cpu_overhead': 0.18
            }
        }
        
        # Network architecture
        self.network_configs = {
            'non-production': {
                'name': 'Non-Production: San Jose → AWS West 2',
                'path': ['San Jose DC', 'AWS West 2'],
                'max_bandwidth': 2000,  # Mbps
                'base_latency': 15,
                'reliability': 99.9,
                'cost_per_gb': 0.09
            },
            'production': {
                'name': 'Production: San Antonio → San Jose → AWS West 2',
                'path': ['San Antonio DC', 'San Jose DC', 'AWS West 2'],
                'max_bandwidth': 10000,  # SA to SJ
                'bottleneck_bandwidth': 2000,  # SJ to AWS
                'base_latency': 35,
                'reliability': 99.95,
                'cost_per_gb': 0.07
            }
        }
        
        # Migration tools
        self.migration_tools = {
            'datasync': {
                'name': 'AWS DataSync',
                'best_for': 'Homogeneous Migrations',
                'efficiency': {'homogeneous': 0.88, 'heterogeneous': 0.45},
                'supports_compression': True,
                'supports_cdc': False
            },
            'dms': {
                'name': 'AWS Database Migration Service',
                'best_for': 'Heterogeneous Migrations',
                'efficiency': {'homogeneous': 0.70, 'heterogeneous': 0.85},
                'supports_compression': False,
                'supports_cdc': True
            }
        }
    
    def calculate_vrops_impact(self, config):
        """Calculate VROPS optimization impact for virtual machines"""
        if not config.get('vrops_enabled', False) or config.get('server_type') != 'virtual':
            return {'cpu_boost': 0, 'memory_boost': 0, 'network_boost': 0, 'overall_boost': 0}
        
        # VROPS optimization based on utilization and optimization score
        cpu_optimization = max(0, (100 - config.get('vrops_cpu_utilization', 65)) / 100 * 0.15)
        memory_optimization = max(0, (100 - config.get('vrops_memory_utilization', 70)) / 100 * 0.12)
        network_optimization = max(0, (100 - config.get('vrops_network_utilization', 45)) / 100 * 0.10)
        
        # Optimization score impact
        score_multiplier = config.get('vrops_optimization_score', 75) / 100
        
        cpu_boost = cpu_optimization * score_multiplier
        memory_boost = memory_optimization * score_multiplier
        network_boost = network_optimization * score_multiplier
        
        overall_boost = (cpu_boost + memory_boost + network_boost) / 3
        
        return {
            'cpu_boost': cpu_boost,
            'memory_boost': memory_boost,
            'network_boost': network_boost,
            'overall_boost': overall_boost
        }
    
    def simulate_network_traffic(self, environment):
        """Simulate real-time network traffic conditions"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Business hours effect (9 AM - 5 PM)
        business_hour_multiplier = 1.4 if 9 <= hour <= 17 else 0.8
        
        # Base congestion with sine wave variation
        time_factor = time.time() / 30000  # Slow variation
        base_congestion = 10 + math.sin(time_factor) * 15 + random.uniform(-5, 5)
        congestion = max(0, min(80, base_congestion * business_hour_multiplier))
        
        # Packet loss increases with congestion
        packet_loss = max(0, (congestion / 100) * 2 + random.uniform(0, 0.5))
        
        # Jitter varies with network conditions
        jitter = 2 + (congestion / 10) + random.uniform(0, 8)
        
        # Available bandwidth decreases with congestion
        network = self.network_configs[environment]
        max_bw = network.get('bottleneck_bandwidth', network['max_bandwidth'])
        available_bandwidth = max_bw * (1 - congestion / 100)
        
        return {
            'congestion': round(congestion),
            'packet_loss': round(packet_loss, 1),
            'jitter': round(jitter, 1),
            'available_bandwidth': round(available_bandwidth)
        }
    
    def calculate_migration_performance(self, config, network_traffic):
        """Calculate comprehensive migration performance"""
        server = self.server_configs[config['server_type']]
        storage = self.storage_configs[config['storage_type']]
        tool = self.migration_tools[config['migration_tool']]
        network = self.network_configs[config['environment']]
        vrops_impact = self.calculate_vrops_impact(config)
        
        # Base performance calculation
        base_latency = network['base_latency'] + network_traffic['jitter']
        effective_bandwidth = network_traffic['available_bandwidth']
        
        # Server efficiency with VROPS optimization
        server_efficiency = server['cpu_efficiency'] * server['memory_efficiency'] * server['io_throughput']
        if config['server_type'] == 'virtual' and config.get('vrops_enabled', False):
            server_efficiency *= (1 + vrops_impact['overall_boost'])
        
        # Storage and OS efficiency
        storage_efficiency = storage['base_efficiency'] * (1.0 if storage['os'] == 'linux' else 0.85)
        
        # Migration tool efficiency
        tool_efficiency = tool['efficiency'][config['database_type']]
        
        # Network quality impact
        network_quality = (1 - network_traffic['packet_loss'] / 100) * (1 - network_traffic['congestion'] / 200)
        
        # Calculate final throughput
        throughput = (effective_bandwidth * 0.7 * server_efficiency * 
                     storage_efficiency * tool_efficiency * network_quality)
        
        # Database size impact (larger databases are slower)
        db_size_impact = max(0.6, 1 - (config['db_size'] / 50000))
        throughput *= db_size_impact
        
        # Compression benefit for DataSync
        if config['migration_tool'] == 'datasync' and config.get('compression_enabled', True):
            throughput *= 1.25
        
        # Add some realistic variation
        throughput *= random.uniform(0.85, 1.15)
        throughput = max(10, throughput)
        
        # Calculate time and cost
        data_to_transfer = config['db_size'] if config['migration_tool'] == 'dms' else config['data_size']
        estimated_time = (data_to_transfer * 8 * 1000) / (throughput * 3600)  # Hours
        
        # Cost calculation
        transfer_cost = data_to_transfer * network['cost_per_gb']
        compute_cost = estimated_time * (45 if config['server_type'] == 'virtual' else 75)
        total_cost = transfer_cost + compute_cost
        
        return {
            'throughput': throughput,
            'latency': base_latency,
            'estimated_time': estimated_time,
            'total_cost': total_cost,
            'network_efficiency': throughput / effective_bandwidth,
            'server_efficiency': server_efficiency,
            'tool_efficiency': tool_efficiency,
            'vrops_impact': vrops_impact['overall_boost'] * 100
        }
    
    def generate_performance_comparison(self, config):
        """Generate performance comparison data for different database sizes"""
        db_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
        comparison_data = []
        
        for size in db_sizes:
            # Physical server performance
            physical_throughput = 1200 * (1 - size / 100000) * 0.9
            
            # Virtual machine without VROPS
            vm_base_throughput = physical_throughput * 0.7
            
            # Virtual machine with VROPS
            vm_vrops_throughput = vm_base_throughput * 1.15
            
            comparison_data.append({
                'Database Size (GB)': size,
                'Physical Server': max(100, physical_throughput),
                'Virtual Machine': max(70, vm_base_throughput),
                'VM + VROPS': max(80, vm_vrops_throughput)
            })
        
        return pd.DataFrame(comparison_data)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ VM vs Physical Server Migration Performance Tool</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Real-time analysis of server types, VROPS optimization, and database migration performance</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">DataSync • DMS • Network Traffic • VROPS Integration • AI-Powered</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_controls():
    """Render sidebar configuration controls"""
    st.sidebar.header("🔧 Configuration Controls")
    
    # Environment and server configuration
    st.sidebar.subheader("🖥️ Infrastructure Setup")
    environment = st.sidebar.selectbox(
        "Environment",
        ["non-production", "production"],
        format_func=lambda x: "Non-Production (SJ→AWS)" if x == "non-production" else "Production (SA→SJ→AWS)"
    )
    
    server_type = st.sidebar.selectbox(
        "Server Type",
        ["physical", "virtual"],
        format_func=lambda x: "Physical Server" if x == "physical" else "Virtual Machine"
    )
    
    storage_type = st.sidebar.selectbox(
        "Storage Type",
        ["nas-linux", "windows-share"],
        format_func=lambda x: "NAS Linux (Red Hat, 48GB)" if x == "nas-linux" else "Windows Share (80GB, VMAX)"
    )
    
    # Migration configuration
    st.sidebar.subheader("🔄 Migration Setup")
    migration_tool = st.sidebar.selectbox(
        "Migration Tool",
        ["datasync", "dms"],
        format_func=lambda x: "DataSync (Homogeneous)" if x == "datasync" else "DMS (Heterogeneous)"
    )
    
    database_type = st.sidebar.selectbox(
        "Database Type",
        ["homogeneous", "heterogeneous"]
    )
    
    data_size = st.sidebar.number_input(
        "Total Data Size (GB)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=100
    )
    
    db_size = st.sidebar.number_input(
        "Database Size (GB)",
        min_value=100,
        max_value=data_size,
        value=min(2000, data_size),
        step=100
    )
    
    compression_enabled = st.sidebar.checkbox("Enable Compression", value=True)
    
    # VROPS Configuration (only for VMs)
    vrops_config = {}
    if server_type == "virtual":
        st.sidebar.subheader("👁️ VROPS Configuration")
        vrops_enabled = st.sidebar.checkbox("Enable VROPS Monitoring", value=False)
        
        if vrops_enabled:
            vrops_config = {
                'vrops_enabled': True,
                'vrops_cpu_utilization': st.sidebar.slider("CPU Utilization (%)", 20, 95, 65),
                'vrops_memory_utilization': st.sidebar.slider("Memory Utilization (%)", 30, 90, 70),
                'vrops_network_utilization': st.sidebar.slider("Network Utilization (%)", 10, 80, 45),
                'vrops_optimization_score': st.sidebar.slider("Optimization Score", 40, 100, 75),
                'vrops_alerts': st.sidebar.number_input("Active Alerts", 0, 10, 2)
            }
        else:
            vrops_config = {'vrops_enabled': False}
    
    # Live updates control
    st.sidebar.subheader("⚡ Real-time Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (2s)", value=True)
    
    if st.sidebar.button("🔄 Manual Refresh"):
        st.rerun()
    
    return {
        'environment': environment,
        'server_type': server_type,
        'storage_type': storage_type,
        'migration_tool': migration_tool,
        'database_type': database_type,
        'data_size': data_size,
        'db_size': db_size,
        'compression_enabled': compression_enabled,
        'auto_refresh': auto_refresh,
        **vrops_config
    }

def render_architecture_visualization(config, calculator):
    """Render architecture visualization"""
    st.subheader("🏗️ Migration Architecture")
    
    server = calculator.server_configs[config['server_type']]
    storage = calculator.storage_configs[config['storage_type']]
    network = calculator.network_configs[config['environment']]
    tool = calculator.migration_tools[config['migration_tool']]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🖥️ Server Configuration</h4>
            <p><strong>Type:</strong> {server['name']}</p>
            <p><strong>CPU Efficiency:</strong> {server['cpu_efficiency']*100:.0f}%</p>
            <p><strong>Memory Efficiency:</strong> {server['memory_efficiency']*100:.0f}%</p>
            <p><strong>I/O Throughput:</strong> {server['io_throughput']*100:.0f}%</p>
            {f"<p style='color: #6f42c1;'><strong>VROPS:</strong> Active</p>" if config.get('vrops_enabled') else ""}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💾 Storage Configuration</h4>
            <p><strong>Type:</strong> {storage['name']}</p>
            <p><strong>Memory:</strong> {storage['memory']}GB RAM</p>
            <p><strong>Adapter:</strong> {storage['ethernet_adapter']}</p>
            <p><strong>OS:</strong> {storage['os'].upper()}</p>
            <p><strong>Efficiency:</strong> {storage['base_efficiency']*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔄 Migration Tool</h4>
            <p><strong>Tool:</strong> {tool['name']}</p>
            <p><strong>Best For:</strong> {tool['best_for']}</p>
            <p><strong>Database Type:</strong> {config['database_type'].title()}</p>
            <p><strong>Efficiency:</strong> {tool['efficiency'][config['database_type']]*100:.0f}%</p>
            <p><strong>Database Size:</strong> {config['db_size']:,} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Network path visualization
    st.markdown("### 🌐 Network Path")
    path_str = " → ".join(network['path'])
    bandwidth_info = f"{network.get('bottleneck_bandwidth', network['max_bandwidth']):,} Mbps"
    if 'bottleneck_bandwidth' in network:
        bandwidth_info += f" (bottleneck at SJ→AWS)"
    
    st.info(f"**Path:** {path_str} | **Bandwidth:** {bandwidth_info} | **Latency:** ~{network['base_latency']}ms")

def render_real_time_metrics(config, calculator):
    """Render real-time performance metrics"""
    st.subheader("📊 Real-Time Performance Metrics")
    
    # Simulate network traffic
    network_traffic = calculator.simulate_network_traffic(config['environment'])
    
    # Calculate performance
    performance = calculator.calculate_migration_performance(config, network_traffic)
    
    # Display main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Throughput",
            f"{performance['throughput']:.0f} Mbps",
            delta=f"{random.uniform(-50, 50):.0f} Mbps"
        )
    
    with col2:
        st.metric(
            "Latency",
            f"{performance['latency']:.1f} ms",
            delta=f"{random.uniform(-2, 2):.1f} ms"
        )
    
    with col3:
        st.metric(
            "Network Efficiency",
            f"{performance['network_efficiency']*100:.1f}%",
            delta=f"{random.uniform(-5, 5):.1f}%"
        )
    
    with col4:
        st.metric(
            "Estimated Time",
            f"{performance['estimated_time']:.1f} hours",
            delta=f"{random.uniform(-1, 1):.1f} hours"
        )
    
    with col5:
        if config.get('vrops_enabled'):
            st.metric(
                "VROPS Boost",
                f"+{performance['vrops_impact']:.1f}%",
                delta=f"{random.uniform(-1, 1):.1f}%"
            )
        else:
            st.metric(
                "Total Cost",
                f"${performance['total_cost']:,.0f}",
                delta=f"${random.uniform(-100, 100):.0f}"
            )
    
    # Network traffic details
    st.markdown("### 🌐 Real-Time Network Traffic")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        congestion_status = "🔴" if network_traffic['congestion'] > 60 else "🟡" if network_traffic['congestion'] > 30 else "🟢"
        st.metric("Network Congestion", f"{network_traffic['congestion']}%", delta=congestion_status)
    
    with col2:
        loss_status = "🔴" if network_traffic['packet_loss'] > 1 else "🟢"
        st.metric("Packet Loss", f"{network_traffic['packet_loss']}%", delta=loss_status)
    
    with col3:
        st.metric("Jitter", f"{network_traffic['jitter']:.1f} ms")
    
    with col4:
        st.metric("Available Bandwidth", f"{network_traffic['available_bandwidth']:,} Mbps")
    
    return performance, network_traffic

def render_performance_chart(calculator, config):
    """Render performance comparison chart"""
    st.subheader("📈 Database Size vs Throughput Analysis")
    
    # Generate comparison data
    comparison_df = calculator.generate_performance_comparison(config)
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=comparison_df['Database Size (GB)'],
        y=comparison_df['Physical Server'],
        mode='lines+markers',
        name='Physical Server',
        line=dict(color='#28a745', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=comparison_df['Database Size (GB)'],
        y=comparison_df['Virtual Machine'],
        mode='lines+markers',
        name='Virtual Machine',
        line=dict(color='#dc3545', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=comparison_df['Database Size (GB)'],
        y=comparison_df['VM + VROPS'],
        mode='lines+markers',
        name='VM + VROPS',
        line=dict(color='#6f42c1', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Performance Impact of Database Size",
        xaxis_title="Database Size (GB)",
        yaxis_title="Throughput (Mbps)",
        hovermode='x',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <h4 style="color: #155724;">Physical Server</h4>
            <p style="color: #155724;">• Best overall performance</p>
            <p style="color: #155724;">• No hypervisor overhead</p>
            <p style="color: #155724;">• Consistent throughput</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #dc3545;">
            <h4 style="color: #721c24;">Virtual Machine</h4>
            <p style="color: #721c24;">• 20-30% performance penalty</p>
            <p style="color: #721c24;">• Hypervisor overhead</p>
            <p style="color: #721c24;">• Resource contention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e2e3f0 0%, #d6d8e5 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #6f42c1;">
            <h4 style="color: #493073;">VM + VROPS</h4>
            <p style="color: #493073;">• 10-15% optimization boost</p>
            <p style="color: #493073;">• Predictive scaling</p>
            <p style="color: #493073;">• Resource optimization</p>
        </div>
        """, unsafe_allow_html=True)

def render_vrops_dashboard(config, calculator):
    """Render VROPS dashboard for virtual machines"""
    if config['server_type'] != 'virtual' or not config.get('vrops_enabled'):
        return
    
    st.subheader("👁️ VROPS Performance Dashboard")
    
    vrops_impact = calculator.calculate_vrops_impact(config)
    
    # VROPS metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Utilization", f"{config['vrops_cpu_utilization']}%")
        st.progress(config['vrops_cpu_utilization'] / 100)
    
    with col2:
        st.metric("Memory Utilization", f"{config['vrops_memory_utilization']}%")
        st.progress(config['vrops_memory_utilization'] / 100)
    
    with col3:
        st.metric("Network Utilization", f"{config['vrops_network_utilization']}%")
        st.progress(config['vrops_network_utilization'] / 100)
    
    with col4:
        st.metric("Optimization Score", f"{config['vrops_optimization_score']}/100")
        st.progress(config['vrops_optimization_score'] / 100)
    
    # VROPS impact analysis
    st.markdown(f"""
    <div class="vrops-panel">
        <h4>🎯 VROPS Optimization Impact</h4>
        <p><strong>CPU Optimization:</strong> +{vrops_impact['cpu_boost']*100:.1f}% performance improvement</p>
        <p><strong>Memory Optimization:</strong> +{vrops_impact['memory_boost']*100:.1f}% efficiency gain</p>
        <p><strong>Network Optimization:</strong> +{vrops_impact['network_boost']*100:.1f}% throughput boost</p>
        <p style="font-weight: bold; color: #6f42c1;"><strong>Overall Performance Boost:</strong> +{vrops_impact['overall_boost']*100:.1f}%</p>
        {f"<p style='color: #dc3545;'><strong>⚠️ Active Alerts:</strong> {config.get('vrops_alerts', 0)} alerts may impact performance</p>" if config.get('vrops_alerts', 0) > 0 else ""}
    </div>
    """, unsafe_allow_html=True)

def render_migration_tool_comparison(config, calculator):
    """Render migration tool comparison"""
    st.subheader("🔧 Migration Tool Analysis")
    
    tools = calculator.migration_tools
    
    col1, col2 = st.columns(2)
    
    with col1:
        datasync = tools['datasync']
        is_optimal = config['migration_tool'] == 'datasync' and config['database_type'] == 'homogeneous'
        border_color = "#28a745" if is_optimal else "#6c757d"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e7f3ff 0%, #cce7ff 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid {border_color};">
            <h4 style="color: #0066cc;">📁 AWS DataSync</h4>
            <p><strong>Best for:</strong> {datasync['best_for']}</p>
            <p><strong>Homogeneous Efficiency:</strong> {datasync['efficiency']['homogeneous']*100:.0f}%</p>
            <p><strong>Heterogeneous Efficiency:</strong> {datasync['efficiency']['heterogeneous']*100:.0f}%</p>
            <p><strong>Supports Compression:</strong> {'Yes' if datasync['supports_compression'] else 'No'}</p>
            <p><strong>Supports CDC:</strong> {'Yes' if datasync['supports_cdc'] else 'No'}</p>
            <p style="font-weight: bold; color: {'#28a745' if is_optimal else '#6c757d'};">
                {'✅ Optimal for current setup' if is_optimal else '⚠️ Not optimal for current setup'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        dms = tools['dms']
        is_optimal = config['migration_tool'] == 'dms' and config['database_type'] == 'heterogeneous'
        border_color = "#28a745" if is_optimal else "#6c757d"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #d4f4dd 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid {border_color};">
            <h4 style="color: #28a745;">🗄️ AWS DMS</h4>
            <p><strong>Best for:</strong> {dms['best_for']}</p>
            <p><strong>Homogeneous Efficiency:</strong> {dms['efficiency']['homogeneous']*100:.0f}%</p>
            <p><strong>Heterogeneous Efficiency:</strong> {dms['efficiency']['heterogeneous']*100:.0f}%</p>
            <p><strong>Supports Compression:</strong> {'Yes' if dms['supports_compression'] else 'No'}</p>
            <p><strong>Supports CDC:</strong> {'Yes' if dms['supports_cdc'] else 'No'}</p>
            <p style="font-weight: bold; color: {'#28a745' if is_optimal else '#6c757d'};">
                {'✅ Optimal for current setup' if is_optimal else '⚠️ Not optimal for current setup'}
            </p>
        </div>
        """, unsafe_allow_html=True)

def generate_ai_recommendations(config, performance, network_traffic, calculator):
    """Generate AI-powered recommendations"""
    server = calculator.server_configs[config['server_type']]
    tool = calculator.migration_tools[config['migration_tool']]
    vrops_impact = calculator.calculate_vrops_impact(config)
    
    recommendations = []
    
    # Server type analysis
    if config['server_type'] == 'physical':
        recommendations.append("✅ **Physical servers provide optimal performance** with 95% CPU efficiency and minimal overhead.")
    else:
        recommendations.append("⚠️ **Virtual machines have 20-30% performance penalty** due to hypervisor overhead.")
        if config.get('vrops_enabled'):
            recommendations.append(f"✅ **VROPS optimization active** providing +{vrops_impact['overall_boost']*100:.1f}% performance improvement.")
        else:
            recommendations.append("🔧 **Enable VROPS** for 10-15% virtual machine performance improvement.")
    
    # Migration tool analysis
    tool_match = (config['migration_tool'] == 'datasync' and config['database_type'] == 'homogeneous') or \
                 (config['migration_tool'] == 'dms' and config['database_type'] == 'heterogeneous')
    
    if tool_match:
        recommendations.append(f"✅ **Optimal tool selection** - {tool['name']} is ideal for {config['database_type']} migrations.")
    else:
        optimal_tool = 'DataSync' if config['database_type'] == 'homogeneous' else 'DMS'
        recommendations.append(f"⚠️ **Tool mismatch** - Consider {optimal_tool} for {config['database_type']} migrations.")
    
    # Network analysis
    if network_traffic['congestion'] > 60:
        recommendations.append("🔴 **High network congestion** detected - schedule migration during off-peak hours.")
    elif network_traffic['congestion'] > 30:
        recommendations.append("🟡 **Medium network congestion** - monitor and consider timing adjustments.")
    else:
        recommendations.append("🟢 **Good network conditions** - optimal time for migration.")
    
    if network_traffic['packet_loss'] > 1:
        recommendations.append("🔴 **High packet loss** may significantly impact performance.")
    
    # Performance optimizations
    if config['migration_tool'] == 'datasync' and not config.get('compression_enabled'):
        recommendations.append("📦 **Enable compression** for 25% DataSync performance improvement.")
    
    if config['db_size'] > 20000:
        recommendations.append("⚠️ **Large database** - consider breaking migration into smaller chunks.")
    
    if performance['network_efficiency'] < 0.5:
        recommendations.append("🔧 **Low network efficiency** - check configuration and bottlenecks.")
    
    return recommendations

def render_ai_recommendations(config, performance, network_traffic, calculator):
    """Render AI recommendations panel"""
    st.subheader("🤖 AI Performance Analysis & Recommendations")
    
    recommendations = generate_ai_recommendations(config, performance, network_traffic, calculator)
    
    recommendation_text = "\n\n".join(recommendations)
    
    st.markdown(f"""
    <div class="ai-insight">
        <h4>🧠 Intelligent Analysis for {config['server_type'].title()} Server Migration</h4>
        {recommendation_text.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Performance summary
    server = calculator.server_configs[config['server_type']]
    tool = calculator.migration_tools[config['migration_tool']]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Server Analysis**
        - Type: {server['name']}
        - CPU Efficiency: {server['cpu_efficiency']*100:.0f}%
        - Memory Efficiency: {server['memory_efficiency']*100:.0f}%
        - Network Overhead: {server['network_overhead']*100:.0f}%
        """)
    
    with col2:
        st.info(f"""
        **Tool Analysis**
        - Tool: {tool['name']}
        - Database Type: {config['database_type'].title()}
        - Efficiency: {tool['efficiency'][config['database_type']]*100:.0f}%
        - Estimated Time: {performance['estimated_time']:.1f} hours
        """)
    
    with col3:
        st.info(f"""
        **Network Analysis**
        - Congestion: {network_traffic['congestion']}%
        - Packet Loss: {network_traffic['packet_loss']}%
        - Available BW: {network_traffic['available_bandwidth']:,} Mbps
        - Quality Score: {performance['network_efficiency']*100:.0f}%
        """)

def main():
    """Main application function"""
    render_header()
    
    # Initialize calculator
    calculator = MigrationPerformanceCalculator()
    
    # Get configuration from sidebar
    config = render_sidebar_controls()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Real-Time Performance", "⚖️ VM vs Physical Analysis", "🎯 VROPS & Optimization"])
    
    with tab1:
        # Architecture visualization
        render_architecture_visualization(config, calculator)
        
        # Real-time metrics
        performance, network_traffic = render_real_time_metrics(config, calculator)
        
        # AI recommendations
        render_ai_recommendations(config, performance, network_traffic, calculator)
    
    with tab2:
        # Performance comparison chart
        render_performance_chart(calculator, config)
        
        # Migration tool comparison
        render_migration_tool_comparison(config, calculator)
        
        # Performance comparison table
        st.subheader("📋 Detailed Performance Comparison")
        
        comparison_data = {
            'Metric': ['CPU Efficiency', 'Memory Efficiency', 'I/O Throughput', 'Network Overhead', 'Overall Performance'],
            'Physical Server': ['95%', '92%', '100%', '5%', 'Best'],
            'Virtual Machine': ['78%', '75%', '70%', '18%', '70% of Physical'],
            'VM + VROPS': ['85-90%', '82-87%', '78-83%', '12-15%', '80-85% of Physical']
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab3:
        # VROPS Dashboard
        render_vrops_dashboard(config, calculator)
        
        # Migration tool comparison
        render_migration_tool_comparison(config, calculator)
        
        # Optimization recommendations
        st.subheader("💡 Performance Optimization Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🖥️ Server Optimization**
            - Physical servers: 25-30% better performance
            - VMs: Enable VROPS for 10-15% boost
            - CPU scheduling optimization
            - Memory balloon driver tuning
            """)
        
        with col2:
            st.markdown("""
            **🌐 Network Optimization**
            - Monitor real-time congestion patterns
            - Schedule during off-peak hours
            - Optimize packet size and window scaling
            - Enable compression when applicable
            """)
        
        with col3:
            st.markdown("""
            **🔧 Tool Optimization**
            - Match tool to migration type
            - DataSync for homogeneous (88% eff)
            - DMS for heterogeneous (85% eff)
            - Enable CDC for minimal downtime
            """)
    
    # Auto-refresh functionality
    if config.get('auto_refresh', False):
        time.sleep(2)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🤖 AI-Powered Performance Analysis • 🖥️ VM vs Physical Comparison • 👁️ VROPS Integration • ☁️ Streamlit Cloud Ready
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()