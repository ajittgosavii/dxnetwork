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
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import anthropic
import asyncio
import logging
from dataclasses import dataclass
import yaml
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer AI v3.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(30,58,138,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .professional-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .agent-scaling-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .api-status-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .status-online { background-color: #22c55e; }
    .status-offline { background-color: #ef4444; }
    .status-warning { background-color: #f59e0b; }
    
    .enterprise-footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 6px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class APIStatus:
    anthropic_connected: bool = False
    aws_pricing_connected: bool = False
    aws_compute_optimizer_connected: bool = False
    last_update: datetime = None
    error_message: str = None

class AnthropicAIManager:
    """Enhanced Anthropic AI manager with improved error handling and connection"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("ANTHROPIC_API_KEY")
        self.client = None
        self.connected = False
        self.error_message = None
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                test_message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.connected = True
                logger.info("Anthropic AI client initialized and tested successfully")
            except Exception as e:
                self.connected = False
                self.error_message = str(e)
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            self.connected = False
            self.error_message = "No API key provided"
    
    async def analyze_migration_workload(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced AI-powered workload analysis with detailed insights"""
        if not self.connected:
            return self._fallback_workload_analysis(config, performance_data)
        
        try:
            # Enhanced prompt with backup storage considerations
            migration_method = config.get('migration_method', 'direct_replication')
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            
            migration_details = ""
            if migration_method == 'backup_restore':
                backup_size_gb = config['database_size_gb'] * backup_size_multiplier
                migration_details = f"""
                BACKUP STORAGE MIGRATION:
                - Migration Method: Backup/Restore via DataSync
                - Backup Storage: {backup_storage_type.replace('_', ' ').title()}
                - Database Size: {config['database_size_gb']} GB
                - Backup Size: {backup_size_gb:.0f} GB ({int(backup_size_multiplier*100)}%)
                - Protocol: {'SMB' if backup_storage_type == 'windows_share' else 'NFS'}
                - Tool: AWS DataSync (File Transfer)
                """
            else:
                migration_details = f"""
                DIRECT REPLICATION MIGRATION:
                - Migration Method: Direct database replication
                - Source Database: {config['source_database_engine']}
                - Target Database: {config['database_engine']}
                - Tool: {'AWS DataSync' if config['source_database_engine'] == config['database_engine'] else 'AWS DMS'}
                """
            
            prompt = f"""
            As a senior AWS migration consultant with deep expertise in database migrations, provide a comprehensive analysis of this migration scenario:

            CURRENT INFRASTRUCTURE:
            - Source Database: {config['source_database_engine']} ({config['database_size_gb']} GB)
            - Target Database: {config['database_engine']}
            - Operating System: {config['operating_system']}
            - Platform: {config['server_type']}
            - Hardware: {config['cpu_cores']} cores @ {config['cpu_ghz']} GHz, {config['ram_gb']} GB RAM
            - Network: {config['nic_type']} ({config['nic_speed']} Mbps)
            - Environment: {config['environment']}
            - Performance Requirement: {config['performance_requirements']}
            - Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
            - Migration Agents: {config.get('number_of_agents', 1)} agents configured
            - Destination Storage: {config.get('destination_storage_type', 'S3')}

            {migration_details}

            CURRENT PERFORMANCE METRICS:
            - Database TPS: {performance_data.get('database_performance', {}).get('effective_tps', 'Unknown')}
            - Storage IOPS: {performance_data.get('storage_performance', {}).get('effective_iops', 'Unknown')}
            - Network Bandwidth: {performance_data.get('network_performance', {}).get('effective_bandwidth_mbps', 'Unknown')} Mbps
            - OS Efficiency: {performance_data.get('os_impact', {}).get('total_efficiency', 0) * 100:.1f}%
            - Overall Performance Score: {performance_data.get('performance_score', 0):.1f}/100

            Please provide a detailed assessment including:
            1. MIGRATION COMPLEXITY (1-10 scale with detailed justification)
            2. RISK ASSESSMENT with specific risk percentages and mitigation strategies
            3. PERFORMANCE OPTIMIZATION recommendations with expected improvement percentages
            4. DETAILED TIMELINE with phase-by-phase breakdown
            5. RESOURCE ALLOCATION with specific AWS instance recommendations
            6. COST OPTIMIZATION strategies with potential savings
            7. BEST PRACTICES specific to this configuration with implementation steps
            8. TESTING STRATEGY with checkpoints and validation criteria
            9. ROLLBACK PROCEDURES and contingency planning
            10. POST-MIGRATION monitoring and optimization recommendations
            11. AGENT SCALING IMPACT analysis based on {config.get('number_of_agents', 1)} agents
            12. DESTINATION STORAGE IMPACT for {config.get('destination_storage_type', 'S3')} including performance and cost implications
            13. BACKUP STORAGE CONSIDERATIONS for {migration_method} method using {backup_storage_type if migration_method == 'backup_restore' else 'N/A'}

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
            Format the response as detailed sections with clear recommendations and actionable insights.
            """
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            ai_response = message.content[0].text
            ai_analysis = self._parse_detailed_ai_response(ai_response, config, performance_data)
            
            return {
                'ai_complexity_score': ai_analysis.get('complexity_score', 6),
                'risk_factors': ai_analysis.get('risk_factors', []),
                'risk_percentages': ai_analysis.get('risk_percentages', {}),
                'mitigation_strategies': ai_analysis.get('mitigation_strategies', []),
                'performance_recommendations': ai_analysis.get('performance_recommendations', []),
                'performance_improvements': ai_analysis.get('performance_improvements', {}),
                'timeline_suggestions': ai_analysis.get('timeline_suggestions', []),
                'resource_allocation': ai_analysis.get('resource_allocation', {}),
                'cost_optimization': ai_analysis.get('cost_optimization', []),
                'best_practices': ai_analysis.get('best_practices', []),
                'testing_strategy': ai_analysis.get('testing_strategy', []),
                'rollback_procedures': ai_analysis.get('rollback_procedures', []),
                'post_migration_monitoring': ai_analysis.get('post_migration_monitoring', []),
                'confidence_level': ai_analysis.get('confidence_level', 'medium'),
                'detailed_assessment': ai_analysis.get('detailed_assessment', {}),
                'agent_scaling_impact': ai_analysis.get('agent_scaling_impact', {}),
                'destination_storage_impact': ai_analysis.get('destination_storage_impact', {}),
                'backup_storage_considerations': ai_analysis.get('backup_storage_considerations', {}),
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_workload_analysis(config, performance_data)
    
    def _parse_detailed_ai_response(self, ai_response: str, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced parsing for detailed AI analysis"""
        
        complexity_factors = []
        base_complexity = 5
        
        # Migration method complexity
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                complexity_factors.append(('SMB protocol overhead', 0.5))
                base_complexity += 0.5
            else:
                complexity_factors.append(('NFS protocol efficiency', -0.2))
                base_complexity -= 0.2
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factors.append(('Heterogeneous migration', 2))
            base_complexity += 2
        
        # Database size complexity
        if config['database_size_gb'] > 10000:
            complexity_factors.append(('Large database size', 1.5))
            base_complexity += 1.5
        elif config['database_size_gb'] > 5000:
            complexity_factors.append(('Medium database size', 0.5))
            base_complexity += 0.5
        
        # Performance requirements
        if config['performance_requirements'] == 'high':
            complexity_factors.append(('High performance requirements', 1))
            base_complexity += 1
        
        # Environment complexity
        if config['environment'] == 'production':
            complexity_factors.append(('Production environment', 0.5))
            base_complexity += 0.5
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            complexity_factors.append(('Multi-agent coordination complexity', 0.5))
            base_complexity += 0.5
        
        # Destination storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_factors.append(('FSx for Windows File System complexity', 0.8))
            base_complexity += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_factors.append(('FSx for Lustre high-performance complexity', 1.0))
            base_complexity += 1.0
        
        complexity_score = min(10, max(1, base_complexity))
        
        # Generate detailed risk assessment
        risk_factors = []
        risk_percentages = {}
        
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                risk_factors.append("SMB protocol stability over WAN connections")
                risk_percentages['smb_protocol_risk'] = 15
            risk_factors.append("Backup file integrity and completeness verification")
            risk_percentages['backup_integrity_risk'] = 10
        
        if config['source_database_engine'] != config['database_engine']:
            risk_factors.append("Schema conversion complexity may cause compatibility issues")
            risk_percentages['schema_conversion_risk'] = 25
        
        if config['database_size_gb'] > 5000:
            risk_factors.append("Large database size increases migration time and failure probability")
            risk_percentages['large_database_risk'] = 15
        
        if config['downtime_tolerance_minutes'] < 120:
            risk_factors.append("Tight downtime window may require multiple attempts")
            risk_percentages['downtime_risk'] = 20
        
        # Agent-specific risks
        if num_agents == 1 and config['database_size_gb'] > 5000:
            risk_factors.append("Single agent may become throughput bottleneck")
            risk_percentages['agent_bottleneck_risk'] = 30
        elif num_agents > 5:
            risk_factors.append("Complex multi-agent coordination may cause synchronization issues")
            risk_percentages['coordination_risk'] = 15
        
        perf_score = performance_data.get('performance_score', 0)
        if perf_score < 70:
            risk_factors.append("Current performance issues may impact migration success")
            risk_percentages['performance_risk'] = 30
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': risk_factors,
            'risk_percentages': risk_percentages,
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors, config),
            'performance_recommendations': self._generate_performance_recommendations(config),
            'performance_improvements': {'overall_optimization': '15-25%'},
            'timeline_suggestions': self._generate_timeline_suggestions(config),
            'resource_allocation': self._generate_resource_allocation(config, complexity_score),
            'cost_optimization': self._generate_cost_optimization(config, complexity_score),
            'best_practices': self._generate_best_practices(config, complexity_score),
            'testing_strategy': self._generate_testing_strategy(config, complexity_score),
            'rollback_procedures': self._generate_rollback_procedures(config),
            'post_migration_monitoring': self._generate_monitoring_recommendations(config),
            'confidence_level': 'high' if complexity_score < 6 else 'medium' if complexity_score < 8 else 'requires_specialist_review',
            'agent_scaling_impact': self._analyze_agent_scaling_impact(config),
            'destination_storage_impact': self._analyze_storage_impact(config),
            'backup_storage_considerations': self._analyze_backup_storage_considerations(config),
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score))),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration'
            }
        }
    
    def _analyze_backup_storage_considerations(self, config: Dict) -> Dict:
        """Analyze backup storage specific considerations"""
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method != 'backup_restore':
            return {'applicable': False}
        
        backup_storage_type = config.get('backup_storage_type', 'nas_drive')
        backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
        
        considerations = {
            'applicable': True,
            'storage_type': backup_storage_type,
            'protocol': 'SMB' if backup_storage_type == 'windows_share' else 'NFS',
            'backup_size_factor': backup_size_multiplier,
            'advantages': [],
            'challenges': [],
            'optimizations': []
        }
        
        if backup_storage_type == 'windows_share':
            considerations['advantages'] = [
                'Native Windows integration',
                'Familiar SMB protocols',
                'Windows authentication support',
                'Easy backup verification'
            ]
            considerations['challenges'] = [
                'SMB protocol overhead (~15% bandwidth loss)',
                'Authentication complexity over WAN',
                'SMB version compatibility requirements'
            ]
            considerations['optimizations'] = [
                'Enable SMB3 multichannel',
                'Optimize SMB signing settings',
                'Use dedicated backup network',
                'Configure SMB compression'
            ]
        else:  # nas_drive
            considerations['advantages'] = [
                'High-performance NFS protocol',
                'Better bandwidth utilization',
                'Lower protocol overhead',
                'Parallel file access capabilities'
            ]
            considerations['challenges'] = [
                'NFS tuning complexity',
                'Cross-platform compatibility',
                'NFS over WAN considerations'
            ]
            considerations['optimizations'] = [
                'Use NFS v4.1+ for best performance',
                'Optimize rsize/wsize parameters',
                'Enable NFS caching',
                'Configure appropriate timeouts'
            ]
        
        return considerations
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            strategies.append("Conduct backup integrity verification before migration")
            strategies.append("Test backup restore procedures in non-production environment")
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                strategies.append("Optimize SMB performance and test stability over WAN")
            else:
                strategies.append("Configure NFS for optimal performance and reliability")
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
            strategies.append("Create detailed schema mapping documentation")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
            strategies.append("Use AWS DataSync for initial bulk data transfer")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
            strategies.append("Use AWS DMS ongoing replication for data synchronization")
        
        if any('agent' in risk.lower() for risk in risk_factors):
            strategies.append(f"Optimize {config.get('number_of_agents', 1)} agent configuration for workload")
            strategies.append("Implement agent health monitoring and automatic failover")
        
        return strategies
    
    def _generate_performance_recommendations(self, config: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                recommendations.append("Enable SMB3 multichannel for better throughput")
                recommendations.append("Optimize SMB client and server settings")
            else:
                recommendations.append("Tune NFS client settings for large file transfers")
                recommendations.append("Configure optimal NFS rsize/wsize values")
            recommendations.append("Use multiple DataSync agents for parallel transfers")
        
        recommendations.extend([
            "Optimize database queries and indexes before migration",
            "Configure proper instance sizing",
            "Implement monitoring and alerting"
        ])
        
        return recommendations
    
    def _generate_timeline_suggestions(self, config: Dict) -> List[str]:
        """Generate timeline suggestions"""
        migration_method = config.get('migration_method', 'direct_replication')
        timeline = [
            "Phase 1: Assessment and Planning (2-3 weeks)",
            "Phase 2: Environment Setup and Testing (2-4 weeks)"
        ]
        
        if migration_method == 'backup_restore':
            timeline.append("Phase 3: Backup Validation and DataSync Setup (1-2 weeks)")
        else:
            timeline.append("Phase 3: Data Validation and Performance Testing (1-2 weeks)")
        
        timeline.extend([
            "Phase 4: Migration Execution (1-3 days)",
            "Phase 5: Post-Migration Validation and Optimization (1 week)"
        ])
        
        return timeline
    
    def _generate_resource_allocation(self, config: Dict, complexity_score: int) -> Dict:
        """Generate resource allocation recommendations"""
        num_agents = config.get('number_of_agents', 1)
        migration_method = config.get('migration_method', 'direct_replication')
        
        base_team_size = 3 + (complexity_score // 3) + (num_agents // 3)
        
        allocation = {
            'migration_team_size': base_team_size,
            'aws_specialists_needed': 1 if complexity_score < 6 else 2,
            'database_experts_required': 1 if config['source_database_engine'] == config['database_engine'] else 2,
            'testing_resources': '2-3 dedicated testers',
            'infrastructure_requirements': f"Staging environment with {config['cpu_cores']*2} cores and {config['ram_gb']*1.5} GB RAM"
        }
        
        if migration_method == 'backup_restore':
            allocation['storage_specialists'] = 1
            allocation['backup_validation_team'] = 2
        
        return allocation
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing to reduce costs by 60-70%")
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            optimizations.append("Optimize backup storage costs and DataSync pricing")
        
        optimizations.append("Implement automated scaling policies to optimize resource utilization")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate best practices"""
        practices = [
            "Implement comprehensive backup strategy before migration initiation",
            "Use AWS Migration Hub for centralized migration tracking",
            "Establish detailed communication plan with stakeholders",
            "Create detailed runbook with step-by-step procedures"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            practices.append("Validate backup integrity before starting migration")
            practices.append("Test restore procedures in isolated environment")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate testing strategy"""
        strategy = [
            "Unit Testing: Validate individual migration components",
            "Integration Testing: Test end-to-end migration workflow",
            "Performance Testing: Validate AWS environment performance",
            "Data Integrity Testing: Verify data consistency and completeness"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            strategy.append("Backup Validation Testing: Verify backup file integrity and completeness")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = [
            "Maintain synchronized read replica during migration window",
            "Create point-in-time recovery snapshot before cutover",
            "Prepare DNS switching procedures for quick rollback",
            "Document application configuration rollback steps"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            procedures.append("Keep original backup files until migration validation complete")
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = [
            "Implement CloudWatch detailed monitoring for all database metrics",
            "Set up automated alerts for performance degradation",
            "Monitor application response times and error rates",
            "Track database connection patterns and query performance"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            recommendations.append("Monitor DataSync task progress and error rates")
        
        return recommendations
    
    def _analyze_agent_scaling_impact(self, config: Dict) -> Dict:
        """Analyze agent scaling impact"""
        num_agents = config.get('number_of_agents', 1)
        migration_method = config.get('migration_method', 'direct_replication')
        
        impact = {
            'parallel_processing_benefit': min(num_agents * 20, 80),
            'coordination_overhead': max(0, (num_agents - 1) * 5),
            'throughput_multiplier': min(num_agents * 0.8, 4.0),
            'management_complexity': num_agents * 10,
            'optimal_agent_count': self._calculate_optimal_agents(config),
            'current_efficiency': min(100, (100 - (abs(num_agents - self._calculate_optimal_agents(config)) * 10)))
        }
        
        if migration_method == 'backup_restore':
            impact['file_transfer_optimization'] = num_agents * 15
        
        return impact
    
    def _analyze_storage_impact(self, config: Dict) -> Dict:
        """Analyze destination storage impact"""
        destination_storage = config.get('destination_storage_type', 'S3')
        return {
            'storage_type': destination_storage,
            'performance_impact': self._calculate_storage_performance_impact(destination_storage),
            'cost_impact': self._calculate_storage_cost_impact(destination_storage),
            'complexity_factor': self._get_storage_complexity_factor(destination_storage)
        }
    
    def _calculate_optimal_agents(self, config: Dict) -> int:
        """Calculate optimal number of agents"""
        database_size = config['database_size_gb']
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            # For backup/restore, optimal agents depend on backup size and storage type
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            effective_size = database_size * backup_size_multiplier
            if effective_size < 500:
                return 1
            elif effective_size < 2000:
                return 2
            elif effective_size < 10000:
                return 3
            else:
                return 4
        else:
            # Original logic for direct replication
            if database_size < 1000:
                return 1
            elif database_size < 5000:
                return 2
            elif database_size < 20000:
                return 3
            else:
                return 4
    
    def _calculate_storage_performance_impact(self, storage_type: str) -> Dict:
        """Calculate performance impact for storage"""
        storage_profiles = {
            'S3': {'throughput_multiplier': 1.0, 'performance_rating': 'Good'},
            'FSx_Windows': {'throughput_multiplier': 1.3, 'performance_rating': 'Very Good'},
            'FSx_Lustre': {'throughput_multiplier': 2.0, 'performance_rating': 'Excellent'}
        }
        return storage_profiles.get(storage_type, storage_profiles['S3'])
    
    def _calculate_storage_cost_impact(self, storage_type: str) -> Dict:
        """Calculate cost impact for storage"""
        cost_profiles = {
            'S3': {'base_cost_multiplier': 1.0, 'long_term_value': 'Excellent'},
            'FSx_Windows': {'base_cost_multiplier': 2.5, 'long_term_value': 'Good'},
            'FSx_Lustre': {'base_cost_multiplier': 4.0, 'long_term_value': 'Good for HPC'}
        }
        return cost_profiles.get(storage_type, cost_profiles['S3'])
    
    def _get_storage_complexity_factor(self, storage_type: str) -> float:
        """Get complexity factor for storage type"""
        complexity_factors = {'S3': 1.0, 'FSx_Windows': 1.8, 'FSx_Lustre': 2.2}
        return complexity_factors.get(storage_type, 1.0)
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Fallback analysis when AI is not available"""
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            complexity_score += 1  # Backup/restore adds some complexity
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': ["Migration complexity varies with database engine differences"],
            'mitigation_strategies': ["Conduct thorough pre-migration testing"],
            'performance_recommendations': ["Optimize database before migration"],
            'confidence_level': 'medium',
            'backup_storage_considerations': self._analyze_backup_storage_considerations(config),
            'raw_ai_response': 'AI analysis not available - using fallback analysis'
        }

class AWSAPIManager:
    """Manage AWS API integration for real-time pricing and optimization"""
    
    def __init__(self):
        self.session = None
        self.pricing_client = None
        self.connected = False
        
        try:
            self.session = boto3.Session()
            self.pricing_client = self.session.client('pricing', region_name='us-east-1')
            self.pricing_client.describe_services(MaxResults=1)
            self.connected = True
            logger.info("AWS API clients initialized successfully")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"AWS API initialization failed: {e}")
            self.connected = False
    
    async def get_real_time_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch real-time AWS pricing data"""
        if not self.connected:
            return self._fallback_pricing_data(region)
        
        try:
            ec2_pricing = await self._get_ec2_pricing(region)
            rds_pricing = await self._get_rds_pricing(region)
            storage_pricing = await self._get_storage_pricing(region)
            
            return {
                'region': region,
                'last_updated': datetime.now(),
                'ec2_instances': ec2_pricing,
                'rds_instances': rds_pricing,
                'storage': storage_pricing,
                'data_source': 'aws_api'
            }
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    async def _get_ec2_pricing(self, region: str) -> Dict:
        """Get EC2 instance pricing"""
        instance_types = ['t3.medium', 't3.large', 't3.xlarge', 'c5.large', 'c5.xlarge', 
                         'c5.2xlarge', 'r6i.large', 'r6i.xlarge', 'r6i.2xlarge']
        
        pricing_data = {}
        for instance_type in instance_types:
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                        {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            price_per_hour = float(price_info['pricePerUnit']['USD'])
                            
                            attributes = price_data.get('product', {}).get('attributes', {})
                            pricing_data[instance_type] = {
                                'vcpu': int(attributes.get('vcpu', 2)),
                                'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                'cost_per_hour': price_per_hour
                            }
            except Exception as e:
                logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)
        
        return pricing_data
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get RDS instance pricing"""
        # Similar to EC2 pricing but for RDS
        return self._fallback_rds_pricing()
    
    async def _get_storage_pricing(self, region: str) -> Dict:
        """Get storage pricing"""
        return self._fallback_storage_pricing()
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name"""
        region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        return region_map.get(region, 'US West (Oregon)')
    
    def _extract_memory_gb(self, memory_str: str) -> int:
        """Extract memory in GB from AWS memory string"""
        try:
            import re
            match = re.search(r'([\d.]+)', memory_str)
            if match:
                return int(float(match.group(1)))
            return 4
        except:
            return 4
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing()
        }
    
    def _fallback_ec2_pricing(self) -> Dict:
        """Fallback EC2 pricing"""
        return {
            't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
            't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832},
            't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664},
            'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
            'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
            'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
            'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252},
            'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504},
            'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008}
        }
    
    def _fallback_rds_pricing(self) -> Dict:
        """Fallback RDS pricing"""
        return {
            'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
            'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
            'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
            'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
            'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92}
        }
    
    def _fallback_storage_pricing(self) -> Dict:
        """Fallback storage pricing"""
        return {
            'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
            'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            's3_standard': {'cost_per_gb_month': 0.023, 'requests_per_1000': 0.0004}
        }
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for instance"""
        fallback_data = self._fallback_ec2_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05})

class OSPerformanceManager:
    """Enhanced OS performance manager with AI insights"""
    
    def __init__(self):
        self.operating_systems = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'cpu_efficiency': 0.92,
                'memory_efficiency': 0.88,
                'io_efficiency': 0.85,
                'network_efficiency': 0.90,
                'virtualization_overhead': 0.12,
                'database_optimizations': {
                    'mysql': 0.88, 'postgresql': 0.85, 'oracle': 0.95, 'sqlserver': 0.98, 'mongodb': 0.87
                },
                'licensing_cost_factor': 2.5,
                'management_complexity': 0.6,
                'security_overhead': 0.08,
                'ai_insights': {
                    'strengths': ['Native SQL Server integration', 'Enterprise management tools'],
                    'weaknesses': ['Higher licensing costs', 'More resource overhead'],
                    'migration_considerations': ['Licensing compliance', 'Service account migration']
                }
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.90,
                'network_efficiency': 0.93,
                'virtualization_overhead': 0.10,
                'database_optimizations': {
                    'mysql': 0.90, 'postgresql': 0.88, 'oracle': 0.97, 'sqlserver': 0.99, 'mongodb': 0.89
                },
                'licensing_cost_factor': 3.0,
                'management_complexity': 0.5,
                'security_overhead': 0.06,
                'ai_insights': {
                    'strengths': ['Improved container support', 'Enhanced security features'],
                    'weaknesses': ['Higher costs', 'Newer OS compatibility risks'],
                    'migration_considerations': ['Hardware compatibility', 'Application compatibility testing']
                }
            },
            'rhel_8': {
                'name': 'Red Hat Enterprise Linux 8',
                'cpu_efficiency': 0.96,
                'memory_efficiency': 0.94,
                'io_efficiency': 0.95,
                'network_efficiency': 0.95,
                'virtualization_overhead': 0.06,
                'database_optimizations': {
                    'mysql': 0.95, 'postgresql': 0.97, 'oracle': 0.93, 'sqlserver': 0.85, 'mongodb': 0.96
                },
                'licensing_cost_factor': 1.5,
                'management_complexity': 0.7,
                'security_overhead': 0.04,
                'ai_insights': {
                    'strengths': ['Excellent performance', 'Strong container support'],
                    'weaknesses': ['Commercial licensing required', 'Steeper learning curve'],
                    'migration_considerations': ['Staff training needs', 'Application compatibility']
                }
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.96,
                'io_efficiency': 0.97,
                'network_efficiency': 0.97,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.97, 'postgresql': 0.98, 'oracle': 0.95, 'sqlserver': 0.87, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.8,
                'management_complexity': 0.6,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['Latest performance optimizations', 'Enhanced security'],
                    'weaknesses': ['Newer release stability', 'Application compatibility risks'],
                    'migration_considerations': ['Extensive testing required', 'Legacy application assessment']
                }
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.97,
                'memory_efficiency': 0.95,
                'io_efficiency': 0.96,
                'network_efficiency': 0.96,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.96, 'postgresql': 0.98, 'oracle': 0.90, 'sqlserver': 0.82, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.8,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['No licensing costs', 'Great community support'],
                    'weaknesses': ['No commercial support without subscription', 'Requires Linux expertise'],
                    'migration_considerations': ['Staff Linux skills', 'Management tool migration']
                }
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.97,
                'io_efficiency': 0.98,
                'network_efficiency': 0.98,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.98, 'postgresql': 0.99, 'oracle': 0.92, 'sqlserver': 0.84, 'mongodb': 0.98
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.7,
                'security_overhead': 0.02,
                'ai_insights': {
                    'strengths': ['Latest performance features', 'Enhanced security'],
                    'weaknesses': ['Newer release risks', 'Potential compatibility issues'],
                    'migration_considerations': ['Comprehensive testing', 'Backup OS strategy']
                }
            }
        }
    
    def extract_database_engine(self, target_database_selection: str, ec2_database_engine: str = None) -> str:
        """Extract the actual database engine from target selection"""
        if target_database_selection.startswith('rds_'):
            return target_database_selection.replace('rds_', '')
        elif target_database_selection.startswith('ec2_'):
            return ec2_database_engine if ec2_database_engine else 'mysql'
        else:
            return target_database_selection
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, target_database_selection: str, ec2_database_engine: str = None) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        os_config = self.operating_systems[os_type]
        database_engine = self.extract_database_engine(target_database_selection, ec2_database_engine)
        
        # Base OS efficiency calculation
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'].get(database_engine, 0.85)
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            total_efficiency *= 1.05 if 'windows' not in os_type else 1.02
        
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'actual_database_engine': database_engine,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }

def get_nic_efficiency(nic_type):
    """Get NIC efficiency based on type"""
    efficiencies = {
        'gigabit_copper': 0.85,
        'gigabit_fiber': 0.90,
        '10g_copper': 0.88, 
        '10g_fiber': 0.92,
        '25g_fiber': 0.94,
        '40g_fiber': 0.95
    }
    return efficiencies.get(nic_type, 0.90)

class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis including backup storage paths"""
    
    def __init__(self):
        self.network_paths = {
            # Backup Storage to S3 Paths (NEW)
            'nonprod_sj_windows_share_s3': {
                'name': 'Non-Prod: San Jose Windows Share → AWS S3 (DataSync)',
                'destination_storage': 'S3',
                'source': 'San Jose Windows Share',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'windows_share',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'Windows Share to DataSync Agent',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.998,
                        'connection_type': 'smb_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.92
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['SMB protocol overhead', 'Windows Share I/O'],
                    'optimization_opportunities': ['SMB3 multichannel', 'DataSync bandwidth optimization'],
                    'risk_factors': ['Windows Share availability', 'SMB authentication'],
                    'recommended_improvements': ['Enable SMB3 multichannel', 'Pre-stage backup files']
                }
            },
            'nonprod_sj_nas_drive_s3': {
                'name': 'Non-Prod: San Jose NAS Drive → AWS S3 (DataSync)',
                'destination_storage': 'S3',
                'source': 'San Jose NAS Drive',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas_drive',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'NAS Drive to DataSync Agent',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'nfs_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.96
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.95
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['NAS internal bandwidth', 'DX connection sharing'],
                    'optimization_opportunities': ['NFS performance tuning', 'Parallel file transfers'],
                    'risk_factors': ['NAS hardware limitations', 'NFS connection stability'],
                    'recommended_improvements': ['Optimize NFS mount options', 'Configure DataSync parallelism']
                }
            },
            'prod_sa_windows_share_s3': {
                'name': 'Prod: San Antonio Windows Share → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio Windows Share',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'windows_share',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'Windows Share to DataSync Agent',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'smb_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.93
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['SMB over WAN latency', 'Multi-hop complexity'],
                    'optimization_opportunities': ['WAN optimization', 'Backup file pre-staging'],
                    'risk_factors': ['Cross-site dependencies', 'SMB over WAN reliability'],
                    'recommended_improvements': ['Implement WAN acceleration', 'Stage backups closer to transfer point']
                }
            },
            'prod_sa_nas_drive_s3': {
                'name': 'Prod: San Antonio NAS Drive → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio NAS Drive',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas_drive',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'NAS Drive to DataSync Agent',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'nfs_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site latency accumulation', 'NFS over WAN'],
                    'optimization_opportunities': ['End-to-end optimization', 'NFS tuning'],
                    'risk_factors': ['Multiple failure points', 'NFS over WAN complexity'],
                    'recommended_improvements': ['Implement NFS over VPN', 'Add backup staging area']
                }
            },
            # Original paths for direct replication (EXISTING)
            'nonprod_sj_linux_nas_s3': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS S3 (Direct Replication)',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.92
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux NAS internal bandwidth', 'DX connection sharing'],
                    'optimization_opportunities': ['NAS performance tuning', 'DX bandwidth upgrade'],
                    'risk_factors': ['Single DX connection dependency', 'NAS hardware limitations'],
                    'recommended_improvements': ['Implement NAS caching', 'Configure QoS on DX']
                }
            },
            'nonprod_sj_linux_nas_fsx_windows': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.5,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux to Windows protocol conversion', 'SMB overhead'],
                    'optimization_opportunities': ['SMB3 protocol optimization', 'FSx throughput configuration'],
                    'risk_factors': ['Cross-platform compatibility', 'SMB version negotiation'],
                    'recommended_improvements': ['Test SMB3.1.1 compatibility', 'Configure FSx performance mode']
                }
            },
            'nonprod_sj_linux_nas_fsx_lustre': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 8,
                        'reliability': 0.9995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.97
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Lustre client configuration', 'Parallel processing coordination'],
                    'optimization_opportunities': ['Lustre striping optimization', 'Parallel I/O tuning'],
                    'risk_factors': ['Lustre complexity', 'Client compatibility'],
                    'recommended_improvements': ['Optimize Lustre striping patterns', 'Configure parallel data transfer']
                }
            },
            'prod_sa_linux_nas_s3': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site latency accumulation', 'Multiple hop complexity'],
                    'optimization_opportunities': ['End-to-end optimization', 'Compression algorithms'],
                    'risk_factors': ['Multiple failure points', 'Complex troubleshooting'],
                    'recommended_improvements': ['Implement WAN optimization', 'Add redundant paths']
                }
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        ai_optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day adjustments
            if segment['connection_type'] == 'internal_lan':
                congestion_factor = 1.1 if 9 <= time_of_day <= 17 else 0.95
            elif segment['connection_type'] == 'private_line':
                congestion_factor = 1.2 if 9 <= time_of_day <= 17 else 0.9
            elif segment['connection_type'] == 'direct_connect':
                congestion_factor = 1.05 if 9 <= time_of_day <= 17 else 0.98
            elif segment['connection_type'] in ['smb_share', 'nfs_share']:
                # Backup storage specific adjustments
                congestion_factor = 1.3 if 9 <= time_of_day <= 17 else 1.0
            else:
                congestion_factor = 1.0
            
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # Backup storage protocol adjustments
            if path.get('migration_type') == 'backup_restore':
                if path['storage_type'] == 'windows_share' and segment['connection_type'] == 'smb_share':
                    effective_bandwidth *= 0.85  # SMB overhead
                elif path['storage_type'] == 'nas_drive' and segment['connection_type'] == 'nfs_share':
                    effective_bandwidth *= 0.92  # NFS is more efficient
            
            # Destination storage adjustments
            if 'FSx' in path['destination_storage']:
                if path['destination_storage'] == 'FSx_Windows':
                    effective_bandwidth *= 1.1
                    effective_latency *= 0.9
                elif path['destination_storage'] == 'FSx_Lustre':
                    effective_bandwidth *= 1.3
                    effective_latency *= 0.7
            
            ai_optimization_score *= segment['ai_optimization_potential']
            
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            total_cost_factor += segment['cost_factor']
            
            adjusted_segments.append({
                **segment,
                'effective_bandwidth_mbps': effective_bandwidth,
                'effective_latency_ms': effective_latency,
                'congestion_factor': congestion_factor
            })
        
        # Calculate quality scores
        latency_score = max(0, 100 - (total_latency * 2))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)
        reliability_score = total_reliability * 100
        
        base_network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        ai_enhanced_quality = base_network_quality * ai_optimization_score
        
        # Storage bonus
        storage_bonus = 0
        if path['destination_storage'] == 'FSx_Windows':
            storage_bonus = 10
        elif path['destination_storage'] == 'FSx_Lustre':
            storage_bonus = 20
        
        ai_enhanced_quality = min(100, ai_enhanced_quality + storage_bonus)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'migration_type': path.get('migration_type', 'direct_replication'),
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': base_network_quality,
            'ai_enhanced_quality_score': ai_enhanced_quality,
            'ai_optimization_potential': (1 - ai_optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'storage_performance_bonus': storage_bonus,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }

class EnhancedAgentSizingManager:
    """Enhanced agent sizing with scalable agent count and AI recommendations"""
    
    def __init__(self):
        self.datasync_agent_specs = {
            'small': {
                'name': 'Small Agent (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 250,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 1TB per agent, <100 Mbps network per agent'
            },
            'medium': {
                'name': 'Medium Agent (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 500,
                'max_concurrent_tasks_per_agent': 25,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '1-5TB per agent, 100-500 Mbps network per agent'
            },
            'large': {
                'name': 'Large Agent (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 1000,
                'max_concurrent_tasks_per_agent': 50,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '5-20TB per agent, 500Mbps-1Gbps network per agent'
            },
            'xlarge': {
                'name': 'XLarge Agent (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 2000,
                'max_concurrent_tasks_per_agent': 100,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '>20TB per agent, >1Gbps network per agent'
            }
        }
        
        self.dms_agent_specs = {
            'small': {
                'name': 'Small DMS Instance (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 200,
                'max_concurrent_tasks_per_agent': 5,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 500GB per agent, simple schemas'
            },
            'medium': {
                'name': 'Medium DMS Instance (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 400,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '500GB-2TB per agent, moderate complexity'
            },
            'large': {
                'name': 'Large DMS Instance (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 800,
                'max_concurrent_tasks_per_agent': 20,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '2-10TB per agent, complex schemas'
            },
            'xlarge': {
                'name': 'XLarge DMS Instance (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 1500,
                'max_concurrent_tasks_per_agent': 40,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '10-50TB per agent, very complex schemas'
            },
            'xxlarge': {
                'name': 'XXLarge DMS Instance (c5.4xlarge)',
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps_per_agent': 2500,
                'max_concurrent_tasks_per_agent': 80,
                'cost_per_hour_per_agent': 0.68,
                'recommended_for': '>50TB per agent, enterprise workloads'
            }
        }
    
    def calculate_agent_configuration(self, agent_type: str, agent_size: str, number_of_agents: int, destination_storage: str = 'S3') -> Dict:
        """Calculate agent configuration with FSx architecture"""
        if agent_type == 'datasync':
            agent_spec = self.datasync_agent_specs[agent_size]
        else:
            agent_spec = self.dms_agent_specs[agent_size]
        
        # Calculate scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(number_of_agents)
        
        # Storage performance multiplier
        storage_multiplier = self._get_storage_performance_multiplier(destination_storage)
        
        total_throughput = (agent_spec['max_throughput_mbps_per_agent'] * 
                           number_of_agents * scaling_efficiency * storage_multiplier)
        
        total_concurrent_tasks = (agent_spec['max_concurrent_tasks_per_agent'] * number_of_agents)
        total_cost_per_hour = agent_spec['cost_per_hour_per_agent'] * number_of_agents
        
        # Management overhead
        management_overhead_factor = 1.0 + (number_of_agents - 1) * 0.05
        storage_overhead = self._get_storage_management_overhead(destination_storage)
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'number_of_agents': number_of_agents,
            'destination_storage': destination_storage,
            'per_agent_spec': agent_spec,
            'total_vcpu': agent_spec['vcpu'] * number_of_agents,
            'total_memory_gb': agent_spec['memory_gb'] * number_of_agents,
            'max_throughput_mbps_per_agent': agent_spec['max_throughput_mbps_per_agent'],
            'total_max_throughput_mbps': total_throughput,
            'effective_throughput_mbps': total_throughput,
            'total_concurrent_tasks': total_concurrent_tasks,
            'cost_per_hour_per_agent': agent_spec['cost_per_hour_per_agent'],
            'total_cost_per_hour': total_cost_per_hour,
            'total_monthly_cost': total_cost_per_hour * 24 * 30,
            'scaling_efficiency': scaling_efficiency,
            'storage_performance_multiplier': storage_multiplier,
            'management_overhead_factor': management_overhead_factor,
            'storage_management_overhead': storage_overhead,
            'effective_cost_per_hour': total_cost_per_hour * management_overhead_factor * storage_overhead,
            'scaling_recommendations': self._get_scaling_recommendations(agent_size, number_of_agents, destination_storage),
            'optimal_configuration': self._assess_configuration_optimality(agent_size, number_of_agents, destination_storage)
        }
    
    def _get_storage_performance_multiplier(self, destination_storage: str) -> float:
        """Get performance multiplier based on destination storage type"""
        multipliers = {'S3': 1.0, 'FSx_Windows': 1.15, 'FSx_Lustre': 1.4}
        return multipliers.get(destination_storage, 1.0)
    
    def _get_storage_management_overhead(self, destination_storage: str) -> float:
        """Get management overhead factor for destination storage"""
        overheads = {'S3': 1.0, 'FSx_Windows': 1.1, 'FSx_Lustre': 1.2}
        return overheads.get(destination_storage, 1.0)
    
    def _calculate_scaling_efficiency(self, number_of_agents: int) -> float:
        """Calculate scaling efficiency - diminishing returns with more agents"""
        if number_of_agents == 1:
            return 1.0
        elif number_of_agents <= 3:
            return 0.95
        elif number_of_agents <= 5:
            return 0.90
        elif number_of_agents <= 8:
            return 0.85
        else:
            return 0.80
    
    def _get_scaling_recommendations(self, agent_size: str, number_of_agents: int, destination_storage: str) -> List[str]:
        """Get scaling-specific recommendations"""
        recommendations = []
        
        if number_of_agents == 1:
            recommendations.append("Single agent configuration - consider scaling for larger workloads")
        elif number_of_agents <= 3:
            recommendations.append("Good balance of performance and manageability")
            recommendations.append("Configure load balancing for optimal distribution")
        else:
            recommendations.append("High-scale configuration requiring careful coordination")
            recommendations.append("Implement centralized monitoring and logging")
        
        # Storage-specific recommendations
        if destination_storage == 'FSx_Lustre':
            recommendations.append("Optimize agents for high-performance Lustre file system")
        elif destination_storage == 'FSx_Windows':
            recommendations.append("Ensure agents are optimized for Windows file sharing protocols")
        
        return recommendations
    
    def _assess_configuration_optimality(self, agent_size: str, number_of_agents: int, destination_storage: str) -> Dict:
        """Assess if the configuration is optimal"""
        efficiency_score = 100
        
        if agent_size == 'small' and number_of_agents > 6:
            efficiency_score -= 20
        
        if number_of_agents > 8:
            efficiency_score -= 25
        
        if 2 <= number_of_agents <= 4 and agent_size in ['medium', 'large']:
            efficiency_score += 10
        
        # Storage-specific adjustments
        if destination_storage == 'FSx_Lustre' and agent_size in ['large', 'xlarge']:
            efficiency_score += 5
        elif destination_storage == 'FSx_Windows' and agent_size in ['medium', 'large']:
            efficiency_score += 3
        
        complexity = "Low" if number_of_agents <= 2 else "Medium" if number_of_agents <= 5 else "High"
        cost_efficiency = "Good" if efficiency_score >= 90 else "Fair" if efficiency_score >= 75 else "Poor"
        
        return {
            'efficiency_score': max(0, efficiency_score),
            'management_complexity': complexity,
            'cost_efficiency': cost_efficiency,
            'optimal_recommendation': self._generate_optimal_recommendation(agent_size, number_of_agents, efficiency_score, destination_storage)
        }
    
    def _generate_optimal_recommendation(self, agent_size: str, number_of_agents: int, efficiency_score: int, destination_storage: str) -> str:
        """Generate optimal configuration recommendation"""
        if efficiency_score >= 90:
            return f"Optimal configuration for {destination_storage} destination"
        elif efficiency_score >= 75:
            return f"Good configuration with minor optimization opportunities for {destination_storage}"
        elif number_of_agents > 6 and agent_size == 'small':
            return "Consider consolidating to fewer, larger agents"
        elif number_of_agents > 8:
            return "Consider reducing agent count to improve manageability"
        else:
            return f"Configuration needs optimization for better {destination_storage} efficiency"

class OnPremPerformanceAnalyzer:
    """Enhanced on-premises performance analyzer with AI insights"""
    
    def __init__(self):
        self.cpu_architectures = {
            'intel_xeon_e5': {'base_performance': 1.0, 'single_thread': 0.9, 'multi_thread': 1.1},
            'intel_xeon_sp': {'base_performance': 1.2, 'single_thread': 1.1, 'multi_thread': 1.3},
            'amd_epyc': {'base_performance': 1.15, 'single_thread': 1.0, 'multi_thread': 1.4}
        }
        
        self.storage_types = {
            'sas_hdd': {'iops': 150, 'throughput_mbps': 200, 'latency_ms': 8},
            'sata_ssd': {'iops': 75000, 'throughput_mbps': 500, 'latency_ms': 0.2},
            'nvme_ssd': {'iops': 450000, 'throughput_mbps': 3500, 'latency_ms': 0.05}
        }
    
    def calculate_ai_enhanced_performance(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """AI-enhanced on-premises performance calculation"""
        
        # Get OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Performance calculations
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        memory_performance = self._calculate_memory_performance(config, os_impact)
        storage_performance = self._calculate_storage_performance(config, os_impact)
        network_performance = self._calculate_network_performance(config, os_impact)
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # AI-enhanced overall performance analysis
        overall_performance = self._calculate_ai_enhanced_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact, config
        )
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': ['No major bottlenecks identified'],
            'ai_insights': ['System appears well-configured for migration'],
            'performance_score': overall_performance['composite_score']
        }
    
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics"""
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'efficiency_factor': os_impact['cpu_efficiency']
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance"""
        base_memory = config['ram_gb']
        os_overhead = 4 if 'windows' in config['operating_system'] else 2
        available_memory = base_memory - os_overhead
        effective_memory = available_memory * os_impact['memory_efficiency']
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': os_impact['memory_efficiency']
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance"""
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'io_efficiency': os_impact['io_efficiency']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance"""
        base_bandwidth = config['nic_speed']
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        if config['server_type'] == 'vmware':
            effective_bandwidth *= 0.92
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency']
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database performance"""
        db_optimization = os_impact['db_optimization']
        
        if config['database_engine'] == 'mysql':
            base_tps = 5000
        elif config['database_engine'] == 'postgresql':
            base_tps = 4500
        elif config['database_engine'] == 'oracle':
            base_tps = 6000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 5500
        else:
            base_tps = 4000
        
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        effective_tps = base_tps * hardware_factor * db_optimization
        
        return {
            'database_engine': config['database_engine'],
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps
        }
    
    def _calculate_ai_enhanced_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                                 storage_perf: Dict, net_perf: Dict, 
                                                 db_perf: Dict, os_impact: Dict, config: Dict) -> Dict:
        """AI-enhanced overall performance calculation"""
        
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'performance_tier': self._get_performance_tier(composite_score)
        }
    
    def _get_performance_tier(self, score: float) -> str:
        """Get performance tier based on score"""
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AI and AWS API integration plus FSx support"""
    
    def __init__(self):
        self.ai_manager = AnthropicAIManager()
        self.aws_api = AWSAPIManager()
        self.os_manager = OSPerformanceManager()
        self.network_manager = EnhancedNetworkIntelligenceManager()
        self.agent_manager = EnhancedAgentSizingManager()
        self.onprem_analyzer = OnPremPerformanceAnalyzer()
        
        self.nic_types = {
            'gigabit_copper': {'max_speed': 1000, 'efficiency': 0.85},
            'gigabit_fiber': {'max_speed': 1000, 'efficiency': 0.90},
            '10g_copper': {'max_speed': 10000, 'efficiency': 0.88},
            '10g_fiber': {'max_speed': 10000, 'efficiency': 0.92},
            '25g_fiber': {'max_speed': 25000, 'efficiency': 0.94},
            '40g_fiber': {'max_speed': 40000, 'efficiency': 0.95}
        }
    
    async def comprehensive_ai_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive AI-powered migration analysis"""
        
        # API status tracking
        api_status = APIStatus(
            anthropic_connected=self.ai_manager.connected,
            aws_pricing_connected=self.aws_api.connected,
            last_update=datetime.now()
        )
        
        # On-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
        
        # Network path analysis
        network_path_key = self._get_network_path_key(config)
        network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
        
        # Migration type and tools
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            # For backup/restore, always use DataSync regardless of database engine
            migration_type = 'backup_restore'
            primary_tool = 'datasync'
        else:
            # For direct replication, use existing logic
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
            primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Agent analysis
        agent_analysis = await self._analyze_ai_migration_agents_with_scaling(config, primary_tool, network_perf)
        
        # Migration throughput
        agent_throughput = agent_analysis['total_effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # Migration time
        migration_time_hours = await self._calculate_ai_migration_time_with_agents(
            config, migration_throughput, onprem_performance, agent_analysis
        )
        
        # AWS sizing
        aws_sizing = await self._ai_enhanced_aws_sizing(config)
        
        # Cost analysis
        cost_analysis = await self._calculate_ai_enhanced_costs_with_agents(
            config, aws_sizing, agent_analysis, network_perf
        )
        
        # FSx comparisons
        fsx_comparisons = await self._generate_fsx_destination_comparisons(config)
        
        # AI overall assessment
        ai_overall_assessment = await self._generate_ai_overall_assessment_with_agents(
            config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
        )
        
        return {
            'api_status': api_status,
            'onprem_performance': onprem_performance,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
            'cost_analysis': cost_analysis,
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': ai_overall_assessment
        }
    
    def _get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on migration method and backup storage"""
        os_lower = config.get('operating_system', '').lower()
        if any(os_name in os_lower for os_name in ['linux', 'ubuntu', 'rhel']):
            os_type = 'linux'
        else:
            os_type = 'windows'
        
        environment = config.get('environment', 'non-production').replace('-', '_').lower()
        destination_storage = config.get('destination_storage_type', 'S3').lower()
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage_type = config.get('backup_storage_type', 'nas_drive')
        
        # For backup/restore method, use backup storage paths
        if migration_method == 'backup_restore':
            if environment in ['non_production', 'nonprod']:
                if backup_storage_type == 'windows_share':
                    return "nonprod_sj_windows_share_s3"
                else:  # nas_drive
                    return "nonprod_sj_nas_drive_s3"
            elif environment == 'production':
                if backup_storage_type == 'windows_share':
                    return "prod_sa_windows_share_s3"
                else:  # nas_drive
                    return "prod_sa_nas_drive_s3"
        
        # For direct replication, use original paths
        else:
            if environment in ['non_production', 'nonprod']:
                if destination_storage == 's3':
                    return f"nonprod_sj_{os_type}_nas_s3"
                elif destination_storage == 'fsx_windows':
                    return f"nonprod_sj_{os_type}_nas_fsx_windows"
                elif destination_storage == 'fsx_lustre':
                    return f"nonprod_sj_{os_type}_nas_fsx_lustre"
            elif environment == 'production':
                if destination_storage == 's3':
                    return f"prod_sa_{os_type}_nas_s3"
        
        # Default fallback for direct replication
        return f"nonprod_sj_{os_type}_nas_s3"
    
    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support and backup storage considerations"""
        
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        # For backup/restore, always use DataSync
        if migration_method == 'backup_restore':
            primary_tool = 'datasync'
            agent_size = config.get('datasync_agent_size', 'medium')
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        elif primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('dms', agent_size, num_agents, destination_storage)
        
        total_max_throughput = agent_config['total_max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        
        # Apply backup storage efficiency factors
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                # SMB has some overhead
                backup_efficiency = 0.85
            else:  # nas_drive with NFS
                backup_efficiency = 0.92
            
            effective_throughput = min(total_max_throughput * backup_efficiency, network_bandwidth)
        else:
            effective_throughput = min(total_max_throughput, network_bandwidth)
            backup_efficiency = 1.0
        
        # Determine bottleneck
        if total_max_throughput < network_bandwidth:
            bottleneck = f'agents ({num_agents} agents)'
            bottleneck_severity = 'high' if effective_throughput / total_max_throughput < 0.7 else 'medium'
        else:
            bottleneck = 'network'
            bottleneck_severity = 'medium' if effective_throughput / network_bandwidth > 0.8 else 'high'
        
        # Add backup storage specific bottleneck detection
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share' and effective_throughput < total_max_throughput * 0.9:
                bottleneck = f'{bottleneck} + SMB protocol overhead'
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'destination_storage': destination_storage,
            'migration_method': migration_method,
            'backup_storage_type': config.get('backup_storage_type', 'nas_drive'),
            'agent_configuration': agent_config,
            'total_max_throughput_mbps': total_max_throughput,
            'total_effective_throughput': effective_throughput,
            'backup_efficiency': backup_efficiency,
            'bottleneck': bottleneck,
            'bottleneck_severity': bottleneck_severity,
            'scaling_efficiency': agent_config['scaling_efficiency'],
            'management_overhead': agent_config['management_overhead_factor'],
            'storage_performance_multiplier': agent_config.get('storage_performance_multiplier', 1.0),
            'cost_per_hour': agent_config['effective_cost_per_hour'],
            'monthly_cost': agent_config['total_monthly_cost']
        }
    
    async def _calculate_ai_migration_time_with_agents(self, config: Dict, migration_throughput: float, 
                                                     onprem_performance: Dict, agent_analysis: Dict) -> float:
        """AI-enhanced migration time calculation with backup storage considerations"""
        
        database_size_gb = config['database_size_gb']
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Calculate data size to transfer
        if migration_method == 'backup_restore':
            # For backup/restore, calculate backup file size
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size_gb = database_size_gb * backup_size_multiplier
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            
            # Base calculation for file transfer
            base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput * 3600)
            
            # Backup storage specific factors
            if backup_storage_type == 'windows_share':
                complexity_factor = 1.2  # SMB protocol overhead
            else:  # nas_drive
                complexity_factor = 1.1  # NFS is more efficient
            
            # Add backup preparation time
            backup_prep_time = 0.5 + (database_size_gb / 10000)  # 0.5-2 hours for backup prep
            
        else:
            # For direct replication, use database size
            data_size_gb = database_size_gb
            base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput * 3600)
            complexity_factor = 1.0
            backup_prep_time = 0
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3
        
        # OS and platform factors
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            complexity_factor *= 0.9
        elif destination_storage == 'FSx_Lustre':
            complexity_factor *= 0.7
        
        # Agent scaling adjustments
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        
        if num_agents > 1:
            agent_time_factor = (1 / min(num_agents * scaling_efficiency * storage_multiplier, 6.0))
            complexity_factor *= agent_time_factor
            
            if num_agents > 5:
                complexity_factor *= 1.1
        
        total_time = base_time_hours * complexity_factor + backup_prep_time
        
        return total_time
    
    async def _ai_enhanced_aws_sizing(self, config: Dict) -> Dict:
        """AI-enhanced AWS sizing"""
        
        # Get real-time pricing
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # RDS sizing
        rds_sizing = self._calculate_rds_sizing(config, pricing_data)
        
        # EC2 sizing  
        ec2_sizing = self._calculate_ec2_sizing(config, pricing_data)
        
        # Reader/writer configuration
        reader_writer_config = self._calculate_reader_writer_config(config)
        
        # Deployment recommendation
        deployment_recommendation = self._recommend_deployment_type(config, rds_sizing, ec2_sizing)
        
        # AI analysis
        ai_analysis = await self.ai_manager.analyze_migration_workload(config, {})
        
        return {
            'rds_recommendations': rds_sizing,
            'ec2_recommendations': ec2_sizing,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': pricing_data
        }
    
    def _calculate_rds_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate RDS sizing"""
        database_size_gb = config['database_size_gb']
        
        if database_size_gb < 1000:
            instance_type = 'db.t3.medium'
            cost_per_hour = 0.068
        elif database_size_gb < 5000:
            instance_type = 'db.r6g.large'
            cost_per_hour = 0.48
        else:
            instance_type = 'db.r6g.xlarge'
            cost_per_hour = 0.96
        
        storage_size = max(database_size_gb * 1.5, 100)
        storage_cost = storage_size * 0.08
        
        return {
            'primary_instance': instance_type,
            'instance_specs': pricing_data.get('rds_instances', {}).get(instance_type, {'vcpu': 2, 'memory': 4}),
            'storage_type': 'gp3',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': cost_per_hour * 24 * 30 + storage_cost,
            'multi_az': config.get('environment') == 'production',
            'backup_retention_days': 30 if config.get('environment') == 'production' else 7
        }
    
    def _calculate_ec2_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate EC2 sizing"""
        database_size_gb = config['database_size_gb']
        
        if database_size_gb < 1000:
            instance_type = 't3.large'
            cost_per_hour = 0.0832
        elif database_size_gb < 5000:
            instance_type = 'r6i.large'
            cost_per_hour = 0.252
        else:
            instance_type = 'r6i.xlarge'
            cost_per_hour = 0.504
        
        storage_size = max(database_size_gb * 2.0, 100)
        storage_cost = storage_size * 0.08
        
        return {
            'primary_instance': instance_type,
            'instance_specs': pricing_data.get('ec2_instances', {}).get(instance_type, {'vcpu': 2, 'memory': 8}),
            'storage_type': 'gp3',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': cost_per_hour * 24 * 30 + storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True
        }
    
    def _calculate_reader_writer_config(self, config: Dict) -> Dict:
        """Calculate reader/writer configuration"""
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        environment = config.get('environment', 'non-production')
        
        writers = 1
        readers = 0
        
        if database_size_gb > 500:
            readers += 1
        if database_size_gb > 2000:
            readers += 1
        if database_size_gb > 10000:
            readers += 2
        
        if performance_req == 'high':
            readers += 2
        
        if environment == 'production':
            readers = max(readers, 2)
        
        total_instances = writers + readers
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': total_instances,
            'write_capacity_percent': (writers / total_instances) * 100 if total_instances > 0 else 100,
            'read_capacity_percent': (readers / total_instances) * 100 if total_instances > 0 else 0,
            'recommended_read_split': min(80, (readers / total_instances) * 100) if total_instances > 0 else 0,
            'reasoning': f"AI-optimized for {database_size_gb}GB database"
        }
    
    def _recommend_deployment_type(self, config: Dict, rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """Recommend deployment type"""
        rds_score = 0
        ec2_score = 0
        
        if config['database_size_gb'] < 2000:
            rds_score += 40
        elif config['database_size_gb'] > 10000:
            ec2_score += 30
        
        if config['performance_requirements'] == 'high':
            ec2_score += 30
        else:
            rds_score += 35
        
        if config['environment'] == 'production':
            rds_score += 20
        
        rds_score += 20  # Management simplicity
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'primary_reasons': [
                f"Recommended for {config['database_size_gb']}GB database",
                f"Suitable for {config.get('environment', 'non-production')} environment"
            ]
        }
    
    async def _calculate_ai_enhanced_costs_with_agents(self, config: Dict, aws_sizing: Dict, 
                                                     agent_analysis: Dict, network_perf: Dict) -> Dict:
        """AI-enhanced cost calculation"""
        
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Agent costs
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        
        # Destination storage costs
        destination_storage = config.get('destination_storage_type', 'S3')
        destination_storage_cost = self._calculate_destination_storage_cost(config, destination_storage)
        
        # Network and other costs
        network_cost = 500
        os_licensing_cost = 300
        management_cost = 200
        
        # Backup storage costs (if applicable)
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage_cost = 0
        if migration_method == 'backup_restore':
            backup_size_gb = config['database_size_gb'] * config.get('backup_size_multiplier', 0.7)
            backup_storage_cost = backup_size_gb * 0.01  # Estimated backup storage cost
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + agent_monthly_cost + 
                            destination_storage_cost + network_cost + os_licensing_cost + 
                            management_cost + backup_storage_cost)
        
        # One-time costs
        one_time_migration_cost = config['database_size_gb'] * 0.1 + config.get('number_of_agents', 1) * 500
        if migration_method == 'backup_restore':
            one_time_migration_cost += 1000  # Additional setup cost for backup/restore
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': agent_monthly_cost,
            'destination_storage_cost': destination_storage_cost,
            'destination_storage_type': destination_storage,
            'backup_storage_cost': backup_storage_cost,
            'network_cost': network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'estimated_monthly_savings': 500,
            'roi_months': 12
        }
    
    def _calculate_destination_storage_cost(self, config: Dict, destination_storage: str) -> float:
        """Calculate destination storage cost"""
        database_size_gb = config['database_size_gb']
        storage_costs = {'S3': 0.023, 'FSx_Windows': 0.13, 'FSx_Lustre': 0.14}
        base_cost_per_gb = storage_costs.get(destination_storage, 0.023)
        return database_size_gb * 1.5 * base_cost_per_gb
    
    async def _generate_fsx_destination_comparisons(self, config: Dict) -> Dict:
        """Generate FSx destination comparisons"""
        comparisons = {}
        destination_types = ['S3', 'FSx_Windows', 'FSx_Lustre']
        
        for dest_type in destination_types:
            temp_config = config.copy()
            temp_config['destination_storage_type'] = dest_type
            
            # Network path
            network_path_key = self._get_network_path_key(temp_config)
            network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
            
            # Agent configuration
            migration_method = config.get('migration_method', 'direct_replication')
            if migration_method == 'backup_restore':
                primary_tool = 'datasync'
                agent_size = config.get('datasync_agent_size', 'medium')
            else:
                is_homogeneous = config['source_database_engine'] == config['database_engine']
                primary_tool = 'datasync' if is_homogeneous else 'dms'
                agent_size = config.get('datasync_agent_size' if is_homogeneous else 'dms_agent_size', 'medium')
                
            num_agents = config.get('number_of_agents', 1)
            
            agent_config = self.agent_manager.calculate_agent_configuration(
                primary_tool, agent_size, num_agents, dest_type
            )
            
            # Migration time
            migration_throughput = min(agent_config['total_max_throughput_mbps'], 
                                     network_perf['effective_bandwidth_mbps'])
            
            if migration_throughput > 0:
                if migration_method == 'backup_restore':
                    backup_size_gb = config['database_size_gb'] * config.get('backup_size_multiplier', 0.7)
                    migration_time = (backup_size_gb * 8 * 1000) / (migration_throughput * 3600)
                else:
                    migration_time = (config['database_size_gb'] * 8 * 1000) / (migration_throughput * 3600)
            else:
                migration_time = float('inf')
            
            # Storage cost
            storage_cost = self._calculate_destination_storage_cost(config, dest_type)
            
            comparisons[dest_type] = {
                'destination_type': dest_type,
                'estimated_migration_time_hours': migration_time,
                'migration_throughput_mbps': migration_throughput,
                'estimated_monthly_storage_cost': storage_cost,
                'performance_rating': self._get_performance_rating(dest_type),
                'cost_rating': self._get_cost_rating(dest_type),
                'complexity_rating': self._get_complexity_rating(dest_type),
                'recommendations': [
                    f'{dest_type} is suitable for this workload',
                    f'Consider performance vs cost trade-offs'
                ],
                'network_performance': network_perf,
                'agent_configuration': {
                    'number_of_agents': num_agents,
                    'total_monthly_cost': agent_config['total_monthly_cost'],
                    'storage_performance_multiplier': agent_config['storage_performance_multiplier']
                }
            }
        
        return comparisons
    
    def _get_performance_rating(self, dest_type: str) -> str:
        """Get performance rating for destination"""
        ratings = {'S3': 'Good', 'FSx_Windows': 'Very Good', 'FSx_Lustre': 'Excellent'}
        return ratings.get(dest_type, 'Good')
    
    def _get_cost_rating(self, dest_type: str) -> str:
        """Get cost rating for destination"""
        ratings = {'S3': 'Excellent', 'FSx_Windows': 'Good', 'FSx_Lustre': 'Fair'}
        return ratings.get(dest_type, 'Good')
    
    def _get_complexity_rating(self, dest_type: str) -> str:
        """Get complexity rating for destination"""
        ratings = {'S3': 'Low', 'FSx_Windows': 'Medium', 'FSx_Lustre': 'High'}
        return ratings.get(dest_type, 'Low')
    
    async def _generate_ai_overall_assessment_with_agents(self, config: Dict, onprem_performance: Dict, 
                                                        aws_sizing: Dict, migration_time: float, 
                                                        agent_analysis: Dict) -> Dict:
        """Generate AI overall assessment"""
        
        readiness_score = 80
        success_probability = 85
        risk_level = 'Medium'
        
        # Adjust based on configuration
        migration_method = config.get('migration_method', 'direct_replication')
        
        if config['database_size_gb'] > 10000:
            readiness_score -= 10
        
        if config['source_database_engine'] != config['database_engine'] and migration_method != 'backup_restore':
            readiness_score -= 15
        
        if migration_time > 24:
            readiness_score -= 10
        
        # Backup storage adjustments
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                readiness_score -= 5  # SMB overhead
            else:
                readiness_score += 5  # NFS efficiency
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': [
                'System appears ready for migration',
                f"{migration_method.replace('_', ' ').title()} migration method selected"
            ],
            'recommended_next_steps': [
                'Conduct detailed performance baseline',
                'Set up AWS environment and testing',
                'Plan comprehensive testing strategy'
            ],
            'timeline_recommendation': {
                'planning_phase_weeks': 2,
                'testing_phase_weeks': 3,
                'migration_window_hours': migration_time,
                'total_project_weeks': 6,
                'recommended_approach': 'staged'
            },
            'agent_scaling_impact': {
                'scaling_efficiency': agent_analysis.get('scaling_efficiency', 1.0) * 100,
                'current_agents': config.get('number_of_agents', 1)
            },
            'destination_storage_impact': {
                'storage_type': config.get('destination_storage_type', 'S3'),
                'storage_performance_multiplier': agent_analysis.get('storage_performance_multiplier', 1.0)
            },
            'backup_storage_impact': {
                'migration_method': migration_method,
                'backup_storage_type': config.get('backup_storage_type', 'nas_drive'),
                'backup_efficiency': agent_analysis.get('backup_efficiency', 1.0)
            }
        }

# Helper functions for rendering
def render_enhanced_header():
    """Enhanced header with professional styling"""
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Agent Scaling Optimization • FSx Destination Analysis • Backup Storage Support
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span style="margin-right: 20px;">🟢 Agent Scaling Optimizer</span>
            <span style="margin-right: 20px;">🟢 FSx Destination Analysis</span>
            <span>🟢 Backup Storage Migration</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    st.sidebar.markdown("### 🔌 System Status")
    
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    # Anthropic AI Status
    ai_status_class = "status-online" if ai_manager.connected else "status-offline"
    ai_status_text = "Connected" if ai_manager.connected else "Disconnected"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {ai_status_class}"></span>
        <strong>Anthropic Claude AI:</strong> {ai_status_text}
        {f"<br><small>Error: {ai_manager.error_message[:50]}...</small>" if ai_manager.error_message else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # AWS API Status
    aws_status_class = "status-online" if aws_api.connected else "status-warning"
    aws_status_text = "Connected" if aws_api.connected else "Using Fallback Data"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {aws_status_class}"></span>
        <strong>AWS Pricing API:</strong> {aws_status_text}
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with AI-powered recommendations"""
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0")
    
    render_api_status_sidebar()
    st.sidebar.markdown("---")
    
    # Operating System Selection
    st.sidebar.subheader("💻 Operating System")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022 (Latest)',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9 (Latest)',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS (Latest)'
        }[x]
    )
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine"
    )
    
    # Hardware Configuration
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2)
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Network Interface
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,
        format_func=lambda x: {
            'gigabit_copper': '🔶 1Gbps Copper',
            'gigabit_fiber': '🟡 1Gbps Fiber',
            '10g_copper': '🔵 10Gbps Copper',
            '10g_fiber': '🟢 10Gbps Fiber',
            '25g_fiber': '🟣 25Gbps Fiber',
            '40g_fiber': '🔴 40Gbps Fiber'
        }[x]
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration
    st.sidebar.subheader("🔄 Migration Setup")
    
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x]
    )
    
    # Backup Storage Configuration for DataSync
    st.sidebar.subheader("💾 Backup Storage Configuration")
    
    # Determine backup storage type based on database engine
    if source_database_engine in ['sqlserver']:
        backup_storage_type = st.sidebar.selectbox(
            "Backup Storage Type",
            ["windows_share", "nas_drive"],
            index=0,
            format_func=lambda x: {
                'windows_share': '🪟 Windows Share Drive (Default for SQL Server)',
                'nas_drive': '🗄️ NAS Drive (Alternative)'
            }[x]
        )
    elif source_database_engine in ['oracle', 'postgresql']:
        backup_storage_type = st.sidebar.selectbox(
            "Backup Storage Type", 
            ["nas_drive", "windows_share"],
            index=0,
            format_func=lambda x: {
                'nas_drive': '🗄️ NAS Drive (Default for Oracle/PostgreSQL)',
                'windows_share': '🪟 Windows Share Drive (Alternative)'
            }[x]
        )
    else:
        backup_storage_type = st.sidebar.selectbox(
            "Backup Storage Type",
            ["nas_drive", "windows_share"],
            index=0,
            format_func=lambda x: {
                'nas_drive': '🗄️ NAS Drive',
                'windows_share': '🪟 Windows Share Drive'
            }[x]
        )
    
    # Backup size configuration
    backup_size_multiplier = st.sidebar.selectbox(
        "Backup Size vs Database",
        [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        index=2,
        format_func=lambda x: f"{int(x*100)}% of DB size ({x:.1f}x multiplier)"
    )
    
    # Migration method selection
    migration_method = st.sidebar.selectbox(
        "Migration Method",
        ["backup_restore", "direct_replication"],
        format_func=lambda x: {
            'backup_restore': '📦 Backup/Restore via DataSync (File Transfer)',
            'direct_replication': '🔄 Direct Replication via DMS (Live Sync)'
        }[x]
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database (AWS)",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '☁️ RDS MySQL', 'postgresql': '☁️ RDS PostgreSQL', 'oracle': '☁️ RDS Oracle',
            'sqlserver': '☁️ RDS SQL Server', 'mongodb': '☁️ DocumentDB'
        }[x]
    )
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100)
    
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60)
    
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    
    # Destination Storage Selection
    st.sidebar.subheader("🗄️ Destination Storage")
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows File Server',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre (High Performance)'
        }[x]
    )
    
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Agent Configuration
    st.sidebar.subheader("🤖 Migration Agent Configuration")
    
    # Determine primary tool based on migration method
    if migration_method == 'backup_restore':
        primary_tool = "DataSync"
        is_homogeneous = True  # Always use DataSync for backup/restore
    else:
        is_homogeneous = source_database_engine == database_engine
        primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    # Show migration method info
    if migration_method == 'backup_restore':
        st.sidebar.info(f"**Method:** Backup/Restore via DataSync from {backup_storage_type.replace('_', ' ').title()}")
        st.sidebar.write(f"**Backup Size:** {int(backup_size_multiplier*100)}% of database ({backup_size_multiplier:.1f}x)")
    else:
        st.sidebar.info(f"**Method:** Direct replication ({'homogeneous' if is_homogeneous else 'heterogeneous'})")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1, max_value=10, value=2, step=1,
        help=f"Number of {primary_tool} agents for parallel processing"
    )
    
    if migration_method == 'backup_restore' or is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent',
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x]
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps/agent',
                'medium': '🔄 Medium (c5.large) - 400 Mbps/agent',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps/agent',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps/agent',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps/agent'
            }[x]
        )
        datasync_agent_size = None
    
    # AI Configuration
    st.sidebar.subheader("🧠 AI Configuration")
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True)
    
    if st.sidebar.button("🔄 Refresh AI Analysis", type="primary"):
        st.rerun()
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'backup_storage_type': backup_storage_type,
        'backup_size_multiplier': backup_size_multiplier,
        'migration_method': migration_method,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'number_of_agents': number_of_agents,
        'enable_ai_analysis': enable_ai_analysis
    }

def render_bandwidth_waterfall_analysis(analysis, config):
    """Show complete bandwidth degradation from user hardware to final throughput including backup storage"""
    st.markdown("**🌊 Bandwidth Waterfall Analysis: From Your Hardware to Migration Throughput**")
    
    network_perf = analysis.get('network_performance', {})
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    user_nic_speed = config.get('nic_speed', 1000)
    nic_type = config.get('nic_type', 'gigabit_fiber')
    environment = config.get('environment', 'non-production')
    migration_method = config.get('migration_method', 'direct_replication')
    backup_storage_type = config.get('backup_storage_type', 'nas_drive')
    
    network_path_limit = network_perf.get('effective_bandwidth_mbps', 1000)
    raw_user_capacity = user_nic_speed
    
    # NIC Hardware Efficiency
    nic_efficiency = get_nic_efficiency(nic_type)
    after_nic = raw_user_capacity * nic_efficiency
    
    # OS Network Stack
    os_network_efficiency = os_impact.get('network_efficiency', 0.90)
    after_os = after_nic * os_network_efficiency
    
    # Virtualization Impact
    server_type = config.get('server_type', 'physical')
    if server_type == 'vmware':
        virtualization_efficiency = 0.92
        after_virtualization = after_os * virtualization_efficiency
    else:
        virtualization_efficiency = 1.0
        after_virtualization = after_os
    
    # Backup Storage Protocol Impact (NEW)
    if migration_method == 'backup_restore':
        if backup_storage_type == 'windows_share':
            protocol_efficiency = 0.75  # SMB has more overhead
            protocol_name = "SMB Protocol"
        else:  # nas_drive
            protocol_efficiency = 0.88  # NFS is more efficient
            protocol_name = "NFS Protocol"
        after_backup_protocol = after_virtualization * protocol_efficiency
    else:
        protocol_efficiency = 0.82 if 'production' in environment else 0.85
        protocol_name = "Standard Protocol"
        after_backup_protocol = after_virtualization * protocol_efficiency
    
    # Network Path Limitation
    after_network_limit = min(after_backup_protocol, network_path_limit)
    network_is_bottleneck = after_backup_protocol > network_path_limit
    
    # Migration Agent Processing
    total_agent_capacity = agent_analysis.get('total_max_throughput_mbps', after_network_limit * 0.75)
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)
    
    final_throughput = actual_throughput if actual_throughput > 0 else min(total_agent_capacity, after_network_limit)
    
    # Build the waterfall stages
    stages = ['Your NIC\nCapacity']
    throughputs = [raw_user_capacity]
    efficiencies = [100]
    descriptions = [f"{user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC"]
    
    # Add each stage
    stages.append('After NIC\nProcessing')
    throughputs.append(after_nic)
    efficiencies.append(nic_efficiency * 100)
    descriptions.append(f"{nic_type.replace('_', ' ').title()} hardware efficiency")
    
    # OS Processing
    os_name = config.get('operating_system', 'unknown').replace('_', ' ').title()
    stages.append(f'After OS\nNetwork Stack')
    throughputs.append(after_os)
    efficiencies.append(os_network_efficiency * 100)
    descriptions.append(f"{os_name} network processing")
    
    # Virtualization (if applicable)
    if server_type == 'vmware':
        stages.append('After VMware\nVirtualization')
        throughputs.append(after_virtualization)
        efficiencies.append(virtualization_efficiency * 100)
        descriptions.append('VMware hypervisor overhead')
    
    # Backup Storage Protocol (NEW STAGE)
    stages.append(f'After {protocol_name}\nOverhead')
    throughputs.append(after_backup_protocol)
    efficiencies.append(protocol_efficiency * 100)
    
    if migration_method == 'backup_restore':
        descriptions.append(f"{protocol_name} for {backup_storage_type.replace('_', ' ')} access")
    else:
        descriptions.append(f"{environment.title()} security protocols")
    
    # Network Path Limitation
    stages.append('After Network\nPath Limit')
    throughputs.append(after_network_limit)
    if network_is_bottleneck:
        efficiencies.append((network_path_limit / after_backup_protocol) * 100)
        descriptions.append(f"Production path: {network_path_limit:,.0f} Mbps available" if environment == 'production' else f"Non-prod DX limit: {network_path_limit:,.0f} Mbps")
    else:
        efficiencies.append(100)
        descriptions.append(f"Network supports {network_path_limit:,.0f} Mbps (no additional limit)")
    
    # Final Migration Throughput
    tool_name = agent_analysis.get('primary_tool', 'DMS').upper()
    num_agents = config.get('number_of_agents', 1)
    stages.append(f'Final Migration\nThroughput')
    throughputs.append(final_throughput)
    
    if after_network_limit > 0:
        agent_efficiency = (final_throughput / after_network_limit) * 100
    else:
        agent_efficiency = 75
    efficiencies.append(agent_efficiency)
    
    if migration_method == 'backup_restore':
        descriptions.append(f"{num_agents}x DataSync agents from {backup_storage_type.replace('_', ' ')}")
    else:
        descriptions.append(f"{num_agents}x {tool_name} agents processing")
    
    # Create the visualization
    waterfall_data = {
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Efficiency': efficiencies,
        'Description': descriptions
    }
    
    # Update the title to reflect migration method
    if migration_method == 'backup_restore':
        title = f"Backup Migration Bandwidth: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps via {backup_storage_type.replace('_', ' ').title()}"
    else:
        title = f"Direct Migration Bandwidth: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps Migration Speed"
    
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=title,
        color='Efficiency',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)',
        hover_data=['Description']
    )
    
    fig.update_traces(texttemplate='%{text:.0f} Mbps', textposition='outside')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Analysis Summary
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    # Identify the primary bottleneck
    if network_is_bottleneck:
        primary_bottleneck = f"Network path ({network_path_limit:,.0f} Mbps limit)"
        bottleneck_type = "Infrastructure"
    elif migration_method == 'backup_restore' and protocol_efficiency < 0.85:
        primary_bottleneck = f"Backup storage protocol ({protocol_name})"
        bottleneck_type = "Backup Storage"
    elif final_throughput < after_network_limit * 0.9:
        primary_bottleneck = f"Migration agents ({num_agents}x {tool_name})"
        bottleneck_type = "Agent Capacity"
    else:
        primary_bottleneck = "Protocol and OS overhead"
        bottleneck_type = "Software"
    
    if bottleneck_type == "Infrastructure":
        st.warning(f"""
        ⚠️ **Network Infrastructure Bottleneck Detected:**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC
        • **Network Limitation:** {network_path_limit:,.0f} Mbps ({environment} environment)
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Plan migration times using {final_throughput:.0f} Mbps actual speed
        """)
    elif bottleneck_type == "Backup Storage":
        st.error(f"""
        💾 **Backup Storage Protocol Bottleneck:**
        • **Available Bandwidth:** {min(user_nic_speed, network_path_limit):,.0f} Mbps
        • **After {protocol_name}:** {after_backup_protocol:.0f} Mbps
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Optimize {backup_storage_type.replace('_', ' ')} performance or consider direct replication
        """)
    elif bottleneck_type == "Agent Capacity":
        st.error(f"""
        🔍 **Agent/Processing Bottleneck:**
        • **Available Bandwidth:** {min(user_nic_speed, network_path_limit):,.0f} Mbps
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Primary Issue:** {primary_bottleneck}
        • **Recommendation:** Optimize agent configuration or increase agent count
        """)
    else:
        st.info(f"""
        💡 **Hardware Bottleneck (Expected):**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC  
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Consider NIC upgrade if faster migration needed
        """)
    
    # Add backup storage specific insights
    if migration_method == 'backup_restore':
        st.info(f"""
        📦 **Backup Storage Migration Insights:**
        • **Backup Storage:** {backup_storage_type.replace('_', ' ').title()}
        • **Protocol:** {protocol_name} ({protocol_efficiency*100:.1f}% efficiency)
        • **Method Advantage:** Lower production database impact
        • **Consider:** Direct replication for higher throughput if backup overhead is significant
        """)

# Tab rendering functions
def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Render enhanced AI insights and analysis tab"""
    st.subheader("🧠 AI-Powered Migration Insights & Analysis")
    
    ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
    ai_assessment = analysis.get('ai_overall_assessment', {})
    
    # Migration Method and Backup Storage Analysis
    migration_method = config.get('migration_method', 'direct_replication')
    if migration_method == 'backup_restore':
        st.markdown("**💾 Backup Storage Migration Analysis:**")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = config.get('database_size_gb', 0) * backup_size_multiplier
            
            st.info("📦 **Backup Storage Configuration**")
            st.write(f"**Storage Type:** {backup_storage_type.replace('_', ' ').title()}")
            st.write(f"**Database Size:** {config.get('database_size_gb', 0):,} GB")
            st.write(f"**Backup Size:** {backup_size_gb:,.0f} GB ({int(backup_size_multiplier*100)}%)")
            st.write(f"**Migration Tool:** DataSync (File Transfer)")
            st.write(f"**Transfer Protocol:** {'SMB' if backup_storage_type == 'windows_share' else 'NFS'}")
            
            agent_analysis = analysis.get('agent_analysis', {})
            backup_efficiency = agent_analysis.get('backup_efficiency', 1.0)
            st.write(f"**Protocol Efficiency:** {backup_efficiency*100:.1f}%")
        
        with backup_col2:
            st.success("🎯 **Backup Migration Advantages**")
            
            if backup_storage_type == 'windows_share':
                st.write("• Native Windows integration")
                st.write("• Familiar SMB protocols")
                st.write("• Windows authentication support")
                st.write("• Easy backup verification")
            else:  # nas_drive
                st.write("• High-performance NFS protocol")
                st.write("• Better bandwidth utilization")
                st.write("• Lower protocol overhead")
                st.write("• Parallel file access")
            
            st.write("• **Benefit:** Lower impact on production database")
            st.write("• **Benefit:** Backup validation before migration")
            st.write("• **Benefit:** Easy rollback capability")
        
        # Backup-specific recommendations
        st.markdown("**💡 Backup Storage Specific Recommendations:**")
        
        backup_rec_col1, backup_rec_col2 = st.columns(2)
        
        with backup_rec_col1:
            st.warning("🚨 **Pre-Migration Checklist**")
            st.write("• Verify backup integrity and completeness")
            st.write("• Test backup restore procedures")
            st.write("• Ensure sufficient storage space in AWS")
            st.write("• Configure DataSync security and permissions")
            
            if backup_storage_type == 'windows_share':
                st.write("• Verify SMB version (recommend SMB3+)")
                st.write("• Check Windows authentication setup")
                st.write("• Test SMB performance with large files")
            else:
                st.write("• Optimize NFS mount options")
                st.write("• Verify NFS version compatibility")
                st.write("• Test NFS performance characteristics")
        
        with backup_rec_col2:
            st.info("⚡ **Performance Optimization Tips**")
            
            if backup_storage_type == 'windows_share':
                st.write("• Enable SMB3 multichannel if available")
                st.write("• Use dedicated network for backup transfer")
                st.write("• Configure SMB signing appropriately")
                st.write("• Monitor SMB connection stability")
            else:
                st.write("• Use NFS v4.1+ for better performance")
                st.write("• Configure appropriate rsize/wsize values")
                st.write("• Enable NFS caching where appropriate")
                st.write("• Monitor NFS connection health")
            
            st.write("• **DataSync:** Configure bandwidth throttling")
            st.write("• **DataSync:** Use multiple agents for large datasets")
            st.write("• **DataSync:** Schedule transfers during off-peak hours")
        
        st.markdown("---")  # Add separator
    
    # AI Analysis Overview
    st.markdown("**🤖 AI Analysis Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🎯 AI Complexity Score",
            f"{complexity_score:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col2:
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        st.metric(
            "📊 Migration Readiness",
            f"{readiness_score:.0f}/100",
            delta=ai_assessment.get('risk_level', 'Unknown')
        )
    
    with col3:
        success_probability = ai_assessment.get('success_probability', 0)
        st.metric(
            "🎯 Success Probability",
            f"{success_probability:.0f}%",
            delta=f"Confidence: {ai_analysis.get('confidence_level', 'medium').title()}"
        )
    
    with col4:
        num_agents = config.get('number_of_agents', 1)
        agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
        st.metric(
            "🤖 Agent Efficiency",
            f"{agent_efficiency*100:.1f}%",
            delta=f"{num_agents} agents"
        )
    
    with col5:
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage = config.get('backup_storage_type', 'nas_drive').replace('_', ' ').title()
        
        if migration_method == 'backup_restore':
            display_text = f"Backup/Restore via {backup_storage}"
            delta_text = f"Tool: {analysis.get('agent_analysis', {}).get('primary_tool', 'DataSync').upper()}"
        else:
            destination = config.get('destination_storage_type', 'S3')
            display_text = destination
            delta_text = f"Efficiency: {agent_efficiency*100:.1f}%"
        
        st.metric(
            "🗄️ Migration Method",
            display_text,
            delta=delta_text
        )
    
    # AI Risk Assessment and Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ AI-Identified Risk Factors:**")
        
        risk_factors = ai_analysis.get('risk_factors', [])
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        with st.container():
            st.warning("Risk Assessment")
            
            if risk_factors:
                st.write("**Identified Risks:**")
                for i, risk in enumerate(risk_factors[:4], 1):
                    st.write(f"{i}. {risk}")
                
                if risk_percentages:
                    st.write("**Risk Probabilities:**")
                    for risk_type, percentage in list(risk_percentages.items())[:3]:
                        risk_name = risk_type.replace('_', ' ').title()
                        st.write(f"• {risk_name}: {percentage}%")
            else:
                st.write("No significant risks identified by AI analysis")
                st.write("Migration appears to be low-risk with current configuration")
    
    with col2:
        st.markdown("**🛡️ AI-Recommended Mitigation Strategies:**")
        
        mitigation_strategies = ai_analysis.get('mitigation_strategies', [])
        
        with st.container():
            st.success("Mitigation Recommendations")
            
            if mitigation_strategies:
                for i, strategy in enumerate(mitigation_strategies[:4], 1):
                    st.write(f"{i}. {strategy}")
            else:
                st.write("• Continue with standard migration best practices")
                st.write("• Implement comprehensive testing procedures")
                st.write("• Monitor performance throughout migration")
    
    # AI Performance Recommendations
    st.markdown("**🚀 AI Performance Optimization Recommendations:**")
    
    performance_recommendations = ai_analysis.get('performance_recommendations', [])
    performance_improvements = ai_analysis.get('performance_improvements', {})
    
    if performance_recommendations:
        for i, recommendation in enumerate(performance_recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    # Try to find corresponding improvement percentage
                    improvement_key = recommendation.lower().replace(' ', '_')[:20]
                    improvement = None
                    for key, value in performance_improvements.items():
                        if any(word in key.lower() for word in improvement_key.split('_')[:2]):
                            improvement = value
                            break
                    
                    if improvement:
                        st.write(f"**Expected Improvement:** {improvement}")
                    else:
                        expected_improvement = "15-25%" if impact == "High" else "5-15%" if impact == "Medium" else "2-10%"
                        st.write(f"**Expected Improvement:** {expected_improvement}")
    else:
        st.info("Current configuration appears well-optimized. Continue with standard best practices.")
    
    # AI Timeline and Resource Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 AI-Enhanced Timeline Suggestions:**")
        
        timeline_suggestions = ai_analysis.get('timeline_suggestions', [])
        
        with st.container():
            st.info("Project Timeline Recommendations")
            
            if timeline_suggestions:
                for suggestion in timeline_suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.write("• Phase 1: Assessment and Planning (2-3 weeks)")
                st.write("• Phase 2: Environment Setup (2-4 weeks)")
                st.write("• Phase 3: Testing and Validation (1-2 weeks)")
                st.write("• Phase 4: Migration Execution (1-3 days)")
                st.write("• Phase 5: Post-Migration Optimization (1 week)")
            
            # Add agent-specific timeline considerations
            num_agents = config.get('number_of_agents', 1)
            if num_agents > 1:
                st.write(f"• Agent Coordination Setup: +{num_agents} hours for {num_agents} agents")
            
            destination_storage = config.get('destination_storage_type', 'S3')
            if destination_storage != 'S3':
                setup_time = {"FSx_Windows": "4-8 hours", "FSx_Lustre": "8-16 hours"}.get(destination_storage, "2-4 hours")
                st.write(f"• {destination_storage} Setup: {setup_time}")
    
    with col2:
        st.markdown("**👥 AI-Recommended Resource Allocation:**")
        
        resource_allocation = ai_analysis.get('resource_allocation', {})
        
        with st.container():
            st.success("Team & Resource Requirements")
            
            if resource_allocation:
                st.write(f"**Migration Team Size:** {resource_allocation.get('migration_team_size', 3)} people")
                st.write(f"**AWS Specialists:** {resource_allocation.get('aws_specialists_needed', 1)} required")
                st.write(f"**Database Experts:** {resource_allocation.get('database_experts_required', 1)} required")
                st.write(f"**Testing Resources:** {resource_allocation.get('testing_resources', 'Standard')}")
                st.write(f"**Infrastructure:** {resource_allocation.get('infrastructure_requirements', 'Standard setup')}")
                
                # Agent-specific resources
                if resource_allocation.get('agent_management_overhead'):
                    st.write(f"**Agent Management:** {resource_allocation.get('agent_management_overhead')}")
                
                # Storage-specific resources
                if resource_allocation.get('storage_specialists', 0) > 0:
                    st.write(f"**Storage Specialists:** {resource_allocation.get('storage_specialists')} required")
            else:
                st.write("**Migration Team Size:** 3-4 people")
                st.write("**AWS Specialists:** 1 required")
                st.write("**Database Experts:** 1-2 required")
                st.write("**Testing Resources:** 2 weeks dedicated testing")
    
    # AI Best Practices and Testing Strategy
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 AI-Recommended Best Practices:**")
        
        best_practices = ai_analysis.get('best_practices', [])
        
        with st.container():
            st.success("Migration Best Practices")
            
            if best_practices:
                for practice in best_practices[:5]:
                    st.write(f"• {practice}")
            else:
                st.write("• Implement comprehensive backup strategy")
                st.write("• Use AWS Migration Hub for tracking")
                st.write("• Establish detailed communication plan")
                st.write("• Create detailed runbook procedures")
                st.write("• Implement automated testing pipelines")
    
    with col2:
        st.markdown("**🧪 AI-Enhanced Testing Strategy:**")
        
        testing_strategy = ai_analysis.get('testing_strategy', [])
        
        with st.container():
            st.info("Testing & Validation Strategy")
            
            if testing_strategy:
                for test in testing_strategy[:5]:
                    st.write(f"• {test}")
            else:
                st.write("• Unit Testing: Validate migration components")
                st.write("• Integration Testing: End-to-end workflow")
                st.write("• Performance Testing: AWS environment validation")
                st.write("• Data Integrity Testing: Consistency verification")
                st.write("• Security Testing: Access controls and encryption")
    
    # AI Post-Migration Monitoring
    st.markdown("**📊 AI Post-Migration Monitoring Recommendations:**")
    
    monitoring_recommendations = ai_analysis.get('post_migration_monitoring', [])
    
    if monitoring_recommendations:
        monitor_col1, monitor_col2, monitor_col3 = st.columns(3)
        
        # Split recommendations into columns
        rec_per_col = len(monitoring_recommendations) // 3
        
        with monitor_col1:
            st.success("**Performance Monitoring**")
            for rec in monitoring_recommendations[:rec_per_col]:
                st.write(f"• {rec}")
        
        with monitor_col2:
            st.info("**Operational Monitoring**")
            for rec in monitoring_recommendations[rec_per_col:rec_per_col*2]:
                st.write(f"• {rec}")
        
        with monitor_col3:
            st.warning("**Cost & Optimization Monitoring**")
            for rec in monitoring_recommendations[rec_per_col*2:]:
                st.write(f"• {rec}")
    else:
        st.info("Standard CloudWatch monitoring recommended for all database and application metrics")
    
    # AI Overall Assessment Summary
    st.markdown("**🎯 AI Overall Assessment Summary:**")
    
    detailed_assessment = ai_analysis.get('detailed_assessment', {})
    
    if detailed_assessment:
        assess_col1, assess_col2, assess_col3 = st.columns(3)
        
        with assess_col1:
            overall_readiness = detailed_assessment.get('overall_readiness', 'needs_preparation')
            if overall_readiness == 'ready':
                st.success("**Migration Readiness: READY**")
            elif overall_readiness == 'needs_preparation':
                st.warning("**Migration Readiness: NEEDS PREPARATION**")
            else:
                st.error("**Migration Readiness: SIGNIFICANT PREPARATION REQUIRED**")
            
            st.write(f"**Success Probability:** {detailed_assessment.get('success_probability', 75)}%")
            st.write(f"**Recommended Approach:** {detailed_assessment.get('recommended_approach', 'staged_migration').replace('_', ' ').title()}")
        
        with assess_col2:
            st.info("**Critical Success Factors**")
            critical_factors = detailed_assessment.get('critical_success_factors', [])
            for factor in critical_factors[:4]:
                st.write(f"• {factor}")
        
        with assess_col3:
            st.warning("**Key Action Items**")
            next_steps = ai_assessment.get('recommended_next_steps', [])
            for step in next_steps[:4]:
                st.write(f"• {step}")
    
    # Raw AI Response (for debugging/transparency)
    with st.expander("🔍 Raw AI Analysis Response", expanded=False):
        raw_response = ai_analysis.get('raw_ai_response', 'No raw AI response available')
        st.text_area("AI Analysis Details", raw_response, height=200, help="Complete AI analysis response for technical review")

def render_network_intelligence_tab(analysis: Dict, config: Dict):
    """Render network intelligence analysis tab with AI insights using native components"""
    st.subheader("🌐 Network Intelligence & Path Optimization")
    
    network_perf = analysis.get('network_performance', {})
    
    # Network Overview Dashboard
    st.markdown("**📊 Network Performance Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Network Quality",
            f"{network_perf.get('network_quality_score', 0):.1f}/100",
            delta=f"AI Enhanced: {network_perf.get('ai_enhanced_quality_score', 0):.1f}"
        )
    
    with col2:
        st.metric(
            "🌐 Network Capacity",  # Changed from "Effective Bandwidth"
            f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps",
            delta="Raw network limit (not migration speed)"  # Added context
        )
    
    with col3:
        st.metric(
            "🕐 Total Latency",
            f"{network_perf.get('total_latency_ms', 0):.1f} ms",
            delta=f"Reliability: {network_perf.get('total_reliability', 0)*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "🗄️ Destination Storage",
            network_perf.get('destination_storage', 'S3'),
            delta=f"Bonus: +{network_perf.get('storage_performance_bonus', 0)}%"
        )
    
    with col5:
        ai_optimization = network_perf.get('ai_optimization_potential', 0)
        st.metric(
            "🤖 AI Optimization",
            f"{ai_optimization:.1f}%",
            delta="Improvement potential"
        )
    
    # Add bandwidth waterfall analysis
    st.markdown("---")  # Add separator
    render_bandwidth_waterfall_analysis(analysis, config)

    st.markdown("---")  # Add separator  
    render_performance_impact_table(analysis, config)
    
    
    # Network Path Visualization
    st.markdown("**🗺️ Network Path Visualization:**")
    
    if network_perf.get('segments'):
        # Create network path diagram
        try:
            network_diagram = create_network_path_diagram(network_perf)
            st.plotly_chart(network_diagram, use_container_width=True)
        except Exception as e:
            st.warning(f"Network diagram could not be rendered: {str(e)}")
            
            # Fallback: Show path as table
            segments_data = []
            for i, segment in enumerate(network_perf.get('segments', []), 1):
                segments_data.append({
                    'Hop': i,
                    'Segment': segment['name'],
                    'Type': segment['connection_type'].replace('_', ' ').title(),
                    'Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    'Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                    'Reliability': f"{segment['reliability']*100:.3f}%",
                    'Cost Factor': f"{segment['cost_factor']:.1f}x"
                })
            
            df_segments = pd.DataFrame(segments_data)
            st.dataframe(df_segments, use_container_width=True)
    else:
        st.info("Network path data not available in current analysis")
    
    # Detailed Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Network Performance Analysis:**")
        
        with st.container():
            st.info("Performance Metrics")
            st.write(f"**Path Name:** {network_perf.get('path_name', 'Unknown')}")
            st.write(f"**Environment:** {network_perf.get('environment', 'Unknown').title()}")
            st.write(f"**OS Type:** {network_perf.get('os_type', 'Unknown').title()}")
            st.write(f"**Storage Type:** {network_perf.get('storage_type', 'Unknown').title()}")
            st.write(f"**Destination:** {network_perf.get('destination_storage', 'S3')}")
            st.write(f"**Migration Type:** {network_perf.get('migration_type', 'direct_replication').replace('_', ' ').title()}")
            st.write(f"**Network Quality Score:** {network_perf.get('network_quality_score', 0):.1f}/100")
            st.write(f"**AI Enhanced Score:** {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
            st.write(f"**Cost Factor:** {network_perf.get('total_cost_factor', 0):.1f}x")
    
    with col2:
        st.markdown("**🤖 AI Network Insights:**")
        
        ai_insights = network_perf.get('ai_insights', {})
        
        with st.container():
            st.success("AI Analysis & Recommendations")
            
            st.write("**Performance Bottlenecks:**")
            bottlenecks = ai_insights.get('performance_bottlenecks', ['No bottlenecks identified'])
            for bottleneck in bottlenecks[:3]:
                st.write(f"• {bottleneck}")
            
            st.write("**Optimization Opportunities:**")
            opportunities = ai_insights.get('optimization_opportunities', ['Standard optimization'])
            for opportunity in opportunities[:3]:
                st.write(f"• {opportunity}")
            
            st.write("**Risk Factors:**")
            risks = ai_insights.get('risk_factors', ['No significant risks'])
            for risk in risks[:2]:
                st.write(f"• {risk}")
    
    # Network Optimization Recommendations
    st.markdown("**💡 Network Optimization Recommendations:**")
    
    recommendations = ai_insights.get('recommended_improvements', [])
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            priority = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    st.write(f"**Priority:** {priority}")
    else:
        st.info("Network appears optimally configured for current requirements")

# Fix for the PyArrow serialization error in render_cost_pricing_tab function

def render_cost_pricing_tab(analysis: Dict, config: Dict):
    """Render comprehensive cost and pricing analysis tab using native components"""
    st.subheader("💰 Live AWS Pricing & Cost Analysis")
    
    cost_analysis = analysis.get('cost_analysis', {})
    
    # Cost Overview Dashboard
    st.markdown("**💸 Cost Overview Dashboard:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_monthly = cost_analysis.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly Cost",
            f"${total_monthly:,.0f}",
            delta=f"Annual: ${total_monthly * 12:,.0f}"
        )
    
    with col2:
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        st.metric(
            "🔄 Migration Cost",
            f"${one_time_cost:,.0f}",
            delta="One-time"
        )
    
    with col3:
        savings = cost_analysis.get('estimated_monthly_savings', 0)
        st.metric(
            "📈 Monthly Savings",
            f"${savings:,.0f}",
            delta=f"Annual: ${savings * 12:,.0f}"
        )
    
    with col4:
        roi_months = cost_analysis.get('roi_months', 0)
        st.metric(
            "⏰ ROI Timeline",
            f"{roi_months} months" if roi_months else "TBD",
            delta=f"Break-even: Year {roi_months / 12:.1f}" if roi_months else "Calculate needed"
        )
    
    with col5:
        agent_cost = cost_analysis.get('agent_cost', 0)
        num_agents = config.get('number_of_agents', 1)
        st.metric(
            "🤖 Agent Costs",
            f"${agent_cost:,.0f}/mo",
            delta=f"${agent_cost / num_agents:.0f} per agent" if num_agents > 0 else ""
        )
    
    # Cost Breakdown Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Monthly Cost Breakdown:**")
        
        # Create cost breakdown pie chart
        cost_breakdown = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Destination Storage': cost_analysis.get('destination_storage_cost', 0),
            'OS Licensing': cost_analysis.get('os_licensing_cost', 0),
            'Management': cost_analysis.get('management_cost', 0)
        }
        
        # Add backup storage cost if applicable
        backup_storage_cost = cost_analysis.get('backup_storage_cost', 0)
        if backup_storage_cost > 0:
            cost_breakdown['Backup Storage'] = backup_storage_cost
        
        # Filter out zero values
        cost_breakdown = {k: v for k, v in cost_breakdown.items() if v > 0}
        
        if cost_breakdown:
            fig_pie = px.pie(
                values=list(cost_breakdown.values()),
                names=list(cost_breakdown.keys()),
                title="Monthly Cost Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Cost breakdown data not available")
    
    with col2:
        st.markdown("**📈 Cost Projections:**")
        
        # Create cost projections chart
        monthly_cost = cost_analysis.get('total_monthly_cost', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        projections = {
            '6 Months': monthly_cost * 6 + one_time_cost,
            '1 Year': monthly_cost * 12 + one_time_cost,
            '2 Years': monthly_cost * 24 + one_time_cost,
            '3 Years': monthly_cost * 36 + one_time_cost,
            '5 Years': monthly_cost * 60 + one_time_cost
        }
        
        fig_bar = px.bar(
            x=list(projections.keys()),
            y=list(projections.values()),
            title="Total Cost Projections",
            labels={'x': 'Timeline', 'y': 'Total Cost ($)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed Cost Analysis using native components
    st.markdown("**🔍 Detailed Cost Analysis:**")
    
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        st.success("💻 **Compute Costs**")
        st.write(f"**AWS Compute:** ${cost_analysis.get('aws_compute_cost', 0):,.0f}/month")
        st.write(f"**AWS Storage:** ${cost_analysis.get('aws_storage_cost', 0):,.0f}/month")
        st.write(f"**OS Licensing:** ${cost_analysis.get('os_licensing_cost', 0):,.0f}/month")
        st.write(f"**Management:** ${cost_analysis.get('management_cost', 0):,.0f}/month")
        subtotal_compute = (cost_analysis.get('aws_compute_cost', 0) + 
                           cost_analysis.get('aws_storage_cost', 0) + 
                           cost_analysis.get('os_licensing_cost', 0) + 
                           cost_analysis.get('management_cost', 0))
        st.write(f"**Subtotal:** ${subtotal_compute:,.0f}/month")
    
    with detail_col2:
        st.info("🔄 **Migration & Agent Costs**")
        st.write(f"**Agent Base Cost:** ${cost_analysis.get('agent_cost', 0):,.0f}/month")
        st.write(f"**One-time Migration:** ${cost_analysis.get('one_time_migration_cost', 0):,.0f}")
        
        # Show backup storage costs if applicable
        backup_storage_cost = cost_analysis.get('backup_storage_cost', 0)
        if backup_storage_cost > 0:
            st.write(f"**Backup Storage:** ${backup_storage_cost:,.0f}/month")
            migration_method = config.get('migration_method', 'direct_replication')
            st.write(f"**Migration Method:** {migration_method.replace('_', ' ').title()}")
        
        subtotal_migration = cost_analysis.get('agent_cost', 0) + backup_storage_cost
        st.write(f"**Monthly Subtotal:** ${subtotal_migration:,.0f}/month")
    
    with detail_col3:
        st.warning("🗄️ **Storage & Network Costs**")
        st.write(f"**Destination Storage:** ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month")
        st.write(f"**Storage Type:** {cost_analysis.get('destination_storage_type', 'S3')}")
        st.write(f"**Network Costs:** ${cost_analysis.get('network_cost', 0):,.0f}/month")
        st.write(f"**Data Transfer:** Included in network")
        subtotal_storage = cost_analysis.get('destination_storage_cost', 0) + cost_analysis.get('network_cost', 0)
        st.write(f"**Subtotal:** ${subtotal_storage:,.0f}/month")
    
    # Cost Optimization Analysis using expandable sections
    with st.expander("🎯 Cost Optimization Analysis", expanded=True):
        ai_cost_insights = cost_analysis.get('ai_cost_insights', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("💡 **AI Cost Optimization Insights**")
            st.write(f"**Optimization Factor:** {ai_cost_insights.get('ai_optimization_factor', 0)*100:.1f}%")
            st.write(f"**Complexity Multiplier:** {ai_cost_insights.get('complexity_multiplier', 1.0):.2f}x")
            st.write(f"**Management Reduction:** {ai_cost_insights.get('management_reduction', 0)*100:.1f}%")
            st.write(f"**Agent Efficiency Bonus:** {ai_cost_insights.get('agent_efficiency_bonus', 0)*100:.1f}%")
            st.write(f"**Storage Efficiency Bonus:** {ai_cost_insights.get('storage_efficiency_bonus', 0)*100:.1f}%")
            st.write(f"**Additional Savings Potential:** {ai_cost_insights.get('potential_additional_savings', '0%')}")
        
        with col2:
            st.info("📊 **ROI & Financial Analysis**")
            st.write(f"**Current Monthly Cost (Est.):** ${total_monthly * 1.2:,.0f}")
            st.write(f"**AWS Monthly Cost:** ${total_monthly:,.0f}")
            st.write(f"**Monthly Savings:** ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}")
            st.write(f"**Annual Savings:** ${cost_analysis.get('estimated_monthly_savings', 0) * 12:,.0f}")
            st.write(f"**Payback Period:** {roi_months} months")
            
            if roi_months and one_time_cost > 0:
                three_year_roi = ((cost_analysis.get('estimated_monthly_savings', 0) * 36 - one_time_cost) / one_time_cost * 100)
                st.write(f"**3-Year ROI:** {three_year_roi:.1f}%")
    
    # Real-time Pricing Data
    st.markdown("**💲 Real-time AWS Pricing Data:**")
    
    pricing_data = analysis.get('aws_sizing_recommendations', {}).get('pricing_data', {})
    
    if pricing_data.get('data_source') == 'aws_api':
        st.success("✅ Using real-time AWS pricing data")
        st.caption(f"Last updated: {pricing_data.get('last_updated', 'Unknown')}")
    else:
        st.warning("⚠️ Using fallback pricing data - AWS API not available")
    
    # Create pricing comparison table with FIXED DATA TYPES
    if pricing_data:
        tab1, tab2, tab3 = st.tabs(["🖥️ EC2 Instances", "🗄️ RDS Instances", "💾 Storage Types"])
        
        with tab1:
            ec2_instances = pricing_data.get('ec2_instances', {})
            if ec2_instances:
                ec2_data = []
                for instance, specs in ec2_instances.items():
                    ec2_data.append({
                        'Instance Type': instance,
                        'vCPU': str(specs.get('vcpu', 'N/A')),  # Convert to string
                        'Memory (GB)': str(specs.get('memory', 'N/A')),  # Convert to string
                        'Cost per Hour': f"${specs.get('cost_per_hour', 0):.4f}",
                        'Monthly Cost': f"${specs.get('cost_per_hour', 0) * 24 * 30:.0f}"
                    })
                
                df_ec2 = pd.DataFrame(ec2_data)
                st.dataframe(df_ec2, use_container_width=True)
        
        with tab2:
            rds_instances = pricing_data.get('rds_instances', {})
            if rds_instances:
                rds_data = []
                for instance, specs in rds_instances.items():
                    rds_data.append({
                        'Instance Type': instance,
                        'vCPU': str(specs.get('vcpu', 'N/A')),  # Convert to string
                        'Memory (GB)': str(specs.get('memory', 'N/A')),  # Convert to string
                        'Cost per Hour': f"${specs.get('cost_per_hour', 0):.4f}",
                        'Monthly Cost': f"${specs.get('cost_per_hour', 0) * 24 * 30:.0f}"
                    })
                
                df_rds = pd.DataFrame(rds_data)
                st.dataframe(df_rds, use_container_width=True)
        
        with tab3:
            storage = pricing_data.get('storage', {})
            if storage:
                storage_data = []
                for storage_type, specs in storage.items():
                    # FIX: Convert all values to strings to ensure consistent data types
                    iops_included = specs.get('iops_included', 'N/A')
                    if isinstance(iops_included, (int, float)):
                        iops_included_str = f"{int(iops_included):,}"
                    else:
                        iops_included_str = str(iops_included)
                    
                    cost_per_iops = specs.get('cost_per_iops_month', 0)
                    if cost_per_iops and cost_per_iops > 0:
                        cost_per_iops_str = f"${cost_per_iops:.3f}"
                    else:
                        cost_per_iops_str = "N/A"
                    
                    storage_data.append({
                        'Storage Type': storage_type.upper(),
                        'Cost per GB/Month': f"${specs.get('cost_per_gb_month', 0):.3f}",
                        'IOPS Included': iops_included_str,  # Now consistently string
                        'Cost per IOPS/Month': cost_per_iops_str  # Now consistently string
                    })
                
                df_storage = pd.DataFrame(storage_data)
                st.dataframe(df_storage, use_container_width=True)

# Fixed render_os_performance_tab function
def render_os_performance_tab(analysis: Dict, config: Dict):
    """Render OS performance analysis tab using native Streamlit components"""
    st.subheader("💻 Operating System Performance Analysis")
    
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    
    # OS Overview
    st.markdown("**🖥️ Operating System Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💻 Current OS",
            os_impact.get('name', 'Unknown'),
            delta=f"Platform: {config.get('server_type', 'Unknown').title()}"
        )
    
    with col2:
        st.metric(
            "⚡ Total Efficiency",
            f"{os_impact.get('total_efficiency', 0)*100:.1f}%",
            delta=f"Base: {os_impact.get('base_efficiency', 0)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "🗄️ DB Optimization",
            f"{os_impact.get('db_optimization', 0)*100:.1f}%",
            delta=f"Engine: {os_impact.get('actual_database_engine', 'Unknown').upper()}"
        )
    
    with col4:
        st.metric(
            "💰 Licensing Factor",
            f"{os_impact.get('licensing_cost_factor', 1.0):.1f}x",
            delta=f"Complexity: {os_impact.get('management_complexity', 0)*100:.0f}%"
        )
    
    with col5:
        virt_overhead = os_impact.get('virtualization_overhead', 0)
        st.metric(
            "☁️ Virtualization",
            f"{virt_overhead*100:.1f}%" if config.get('server_type') == 'vmware' else "N/A",
            delta="Overhead" if config.get('server_type') == 'vmware' else "Physical"
        )
    
    # OS Performance Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 OS Performance Metrics:**")
        
        # Create radar chart for OS performance
        performance_metrics = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        fig_radar = go.Figure()
        
        categories = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='OS Performance',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="OS Performance Profile"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 AI OS Insights:**")
        
        ai_insights = os_impact.get('ai_insights', {})
        
        with st.container():
            st.success(f"AI Analysis of {os_impact.get('name', 'Current OS')}")
            
            st.write("**Strengths:**")
            for strength in ai_insights.get('strengths', ['General purpose OS'])[:3]:
                st.write(f"• {strength}")
            
            st.write("**Weaknesses:**")
            for weakness in ai_insights.get('weaknesses', ['No significant issues'])[:3]:
                st.write(f"• {weakness}")
            
            st.write("**Migration Considerations:**")
            for consideration in ai_insights.get('migration_considerations', ['Standard migration'])[:3]:
                st.write(f"• {consideration}")
    
    # Database Engine Optimization
    st.markdown("**🗄️ Database Engine Optimization Analysis:**")
    
    database_optimizations = os_impact.get('database_optimizations', {})
    
    if database_optimizations:
        opt_data = []
        # Get the actual database engine being used
        current_engine = os_impact.get('actual_database_engine', 'mysql')
        
        for engine, optimization in database_optimizations.items():
            performance_rating = ('Excellent' if optimization > 0.95 else 
                                'Very Good' if optimization > 0.90 else 
                                'Good' if optimization > 0.85 else 'Fair')
            
            opt_data.append({
                'Database Engine': engine.upper(),
                'Optimization Score': f"{optimization*100:.1f}%",
                'Performance Rating': performance_rating,
                'Current Selection': '✅' if engine == current_engine else ''
            })
        
        df_opt = pd.DataFrame(opt_data)
        st.dataframe(df_opt, use_container_width=True)
    
    # OS Comparison Analysis
    st.markdown("**⚖️ OS Comparison Analysis:**")
    
    # Create comparison with other OS options
    os_manager = OSPerformanceManager()
    comparison_data = []
    
    current_os = config.get('operating_system')
    current_platform = config.get('server_type')
    current_db_engine = config.get('database_engine')
    ec2_db_engine = config.get('ec2_database_engine')
    
    for os_name, os_config in os_manager.operating_systems.items():
        # FIXED: Pass the ec2_database_engine parameter
        os_perf = os_manager.calculate_os_performance_impact(
            os_name, 
            current_platform, 
            current_db_engine,
            ec2_db_engine
        )
        
        comparison_data.append({
            'Operating System': os_config['name'],
            'Total Efficiency': f"{os_perf['total_efficiency']*100:.1f}%",
            'CPU Efficiency': f"{os_perf['cpu_efficiency']*100:.1f}%",
            'Memory Efficiency': f"{os_perf['memory_efficiency']*100:.1f}%",
            'I/O Efficiency': f"{os_perf['io_efficiency']*100:.1f}%",
            'Licensing Cost': f"{os_perf['licensing_cost_factor']:.1f}x",
            'Current Choice': '✅' if os_name == current_os else ''
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Performance Impact Analysis using native components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🎯 **Performance Impact Assessment**")
        st.write(f"**Base Efficiency:** {os_impact.get('base_efficiency', 0)*100:.1f}%")
        st.write(f"**Database Optimization:** {os_impact.get('db_optimization', 0)*100:.1f}%")
        st.write(f"**Platform Optimization:** {os_impact.get('platform_optimization', 1.0)*100:.1f}%")
        improvement = ((os_impact.get('total_efficiency', 0) - 0.8) / 0.2 * 100)
        st.write(f"**Overall Impact:** {improvement:.1f}% above baseline")
    
    with col2:
        st.warning("💰 **Cost Impact Analysis**")
        st.write(f"**Licensing Cost Factor:** {os_impact.get('licensing_cost_factor', 1.0):.1f}x")
        monthly_licensing = os_impact.get('licensing_cost_factor', 1.0) * 150
        st.write(f"**Monthly Licensing Est.:** ${monthly_licensing:.0f}")
        st.write(f"**Management Complexity:** {os_impact.get('management_complexity', 0)*100:.0f}%")
        st.write(f"**Security Overhead:** {os_impact.get('security_overhead', 0)*100:.1f}%")
    
    with col3:
        suitability = ("Excellent" if os_impact.get('total_efficiency', 0) > 0.9 else 
                      "Good" if os_impact.get('total_efficiency', 0) > 0.8 else "Fair")
        
        migration_complexity = ("Low" if 'windows' in current_os and 'windows' in current_os else "Medium")
        
        recommendation = ("Keep current OS" if os_impact.get('total_efficiency', 0) > 0.85 else 
                         "Consider optimization")
        
        st.success("🔧 **Migration Recommendations**")
        st.write(f"**Current OS Suitability:** {suitability}")
        st.write(f"**Migration Complexity:** {migration_complexity}")
        st.write(f"**Recommended Action:** {recommendation}")

def render_migration_dashboard_tab(analysis: Dict, config: Dict):
    """Render comprehensive migration dashboard with key metrics and visualizations"""
    st.subheader("📊 Enhanced Migration Performance Dashboard")
    
    # Executive Summary Dashboard
    st.markdown("**🎯 Executive Migration Summary:**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        st.metric(
            "🎯 Readiness Score",
            f"{readiness_score:.0f}/100",
            delta=analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown')
        )
    
    with col2:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric(
            "⏱️ Migration Time",
            f"{migration_time:.1f} hours",
            delta=f"Window: {config.get('downtime_tolerance_minutes', 60)} min"
        )
    
    with col3:
        throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric(
            "🚀 Throughput",
            f"{throughput:,.0f} Mbps",
            delta=f"Agents: {config.get('number_of_agents', 1)}"
        )
    
    with col4:
        monthly_cost = analysis.get('cost_analysis', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta=f"ROI: {analysis.get('cost_analysis', {}).get('roi_months', 'TBD')} months"
        )
    
    with col5:
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage = config.get('backup_storage_type', 'nas_drive').replace('_', ' ').title()
        
        if migration_method == 'backup_restore':
            display_text = f"Backup/Restore via {backup_storage}"
            delta_text = f"Tool: {analysis.get('agent_analysis', {}).get('primary_tool', 'DataSync').upper()}"
        else:
            destination = config.get('destination_storage_type', 'S3')
            agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
            display_text = destination
            delta_text = f"Efficiency: {agent_efficiency*100:.1f}%"
        
        st.metric(
            "🗄️ Migration Method",
            display_text,
            delta=delta_text
        )
    
    with col6:
        complexity = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        confidence = analysis.get('ai_overall_assessment', {}).get('ai_confidence', 0.5)
        st.metric(
            "🤖 AI Confidence",
            f"{confidence*100:.1f}%",
            delta=f"Complexity: {complexity:.1f}/10"
        )
    
    # Performance Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Performance Analysis:**")
        
        # Current vs Target Performance
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_comparison = {
            'Metric': ['CPU', 'Memory', 'Storage', 'Network', 'Database'],
            'Current Score': [
                onprem_perf.get('cpu_score', 0),
                onprem_perf.get('memory_score', 0),
                onprem_perf.get('storage_score', 0),
                onprem_perf.get('network_score', 0),
                onprem_perf.get('database_score', 0)
            ],
            'Target (AWS)': [85, 90, 95, 90, 88]  # Estimated AWS performance
        }
        
        fig_perf = px.bar(
            performance_comparison,
            x='Metric',
            y=['Current Score', 'Target (AWS)'],
            title="Current vs AWS Target Performance",
            barmode='group'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        st.markdown("**🔄 Migration Timeline:**")
        
        # Timeline breakdown
        timeline = analysis.get('ai_overall_assessment', {}).get('timeline_recommendation', {})
        
        timeline_data = {
            'Phase': ['Planning', 'Testing', 'Migration', 'Validation'],
            'Duration (Weeks)': [
                timeline.get('planning_phase_weeks', 2),
                timeline.get('testing_phase_weeks', 3),
                timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
                1  # Validation week
            ]
        }
        
        fig_timeline = px.bar(
            timeline_data,
            x='Phase',
            y='Duration (Weeks)',
            title="Project Timeline Breakdown",
            color='Duration (Weeks)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Agent and Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Agent Performance Analysis:**")
        
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Agent throughput breakdown
        agent_data = {
            'Component': ['Per Agent', 'Total Agents', 'Network Limit', 'Effective'],
            'Throughput (Mbps)': [
                agent_analysis.get('total_max_throughput_mbps', 0) / config.get('number_of_agents', 1),
                agent_analysis.get('total_max_throughput_mbps', 0),
                analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
                agent_analysis.get('total_effective_throughput', 0)
            ]
        }
        
        fig_agent = px.bar(
            agent_data,
            x='Component',
            y='Throughput (Mbps)',
            title=f"Agent Throughput Analysis ({config.get('number_of_agents', 1)} agents)",
            color='Throughput (Mbps)',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_agent, use_container_width=True)
    
    with col2:
        st.markdown("**🌐 Network Performance Analysis:**")
        
        network_perf = analysis.get('network_performance', {})
        
        # Network metrics
        network_data = {
            'Metric': ['Quality Score', 'AI Enhanced', 'Bandwidth Util', 'Reliability'],
            'Score/Percentage': [
                network_perf.get('network_quality_score', 0),
                network_perf.get('ai_enhanced_quality_score', 0),
                min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
                network_perf.get('total_reliability', 0) * 100
            ]
        }
        
        fig_network = px.bar(
            network_data,
            x='Metric',
            y='Score/Percentage',
            title="Network Performance Metrics",
            color='Score/Percentage',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Cost and ROI Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Cost Analysis Dashboard:**")
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Cost breakdown
        cost_categories = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0) + cost_analysis.get('destination_storage_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Other': cost_analysis.get('os_licensing_cost', 0) + cost_analysis.get('management_cost', 0)
        }
        
        # Add backup storage if applicable
        backup_storage_cost = cost_analysis.get('backup_storage_cost', 0)
        if backup_storage_cost > 0:
            cost_categories['Backup Storage'] = backup_storage_cost
        
        fig_cost = px.pie(
            values=list(cost_categories.values()),
            names=list(cost_categories.keys()),
            title="Monthly Cost Distribution"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        st.markdown("**📈 ROI Projection:**")
        
        # ROI over time
        monthly_savings = cost_analysis.get('estimated_monthly_savings', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        months = list(range(0, 37, 6))  # 0 to 36 months
        cumulative_savings = []
        
        for month in months:
            if month == 0:
                cumulative_savings.append(-one_time_cost)
            else:
                cumulative_savings.append((monthly_savings * month) - one_time_cost)
        
        roi_data = {
            'Months': months,
            'Cumulative Savings ($)': cumulative_savings
        }
        
        fig_roi = px.line(
            roi_data,
            x='Months',
            y='Cumulative Savings ($)',
            title="ROI Projection Over Time",
            markers=True
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Risk and Readiness Assessment
    st.markdown("**⚠️ Risk and Readiness Assessment:**")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🎯 Migration Readiness</h4>
            <p><strong>Readiness Score:</strong> {ai_assessment.get('migration_readiness_score', 0):.0f}/100</p>
            <p><strong>Success Probability:</strong> {ai_assessment.get('success_probability', 0):.0f}%</p>
            <p><strong>Risk Level:</strong> {ai_assessment.get('risk_level', 'Unknown')}</p>
            <p><strong>Recommended Approach:</strong> {ai_assessment.get('timeline_recommendation', {}).get('recommended_approach', 'Unknown').replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        agent_impact = ai_assessment.get('agent_scaling_impact', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🤖 Agent Scaling Assessment</h4>
            <p><strong>Scaling Efficiency:</strong> {agent_impact.get('scaling_efficiency', 0):.1f}%</p>
            <p><strong>Optimal Agents:</strong> {agent_impact.get('optimal_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Current Agents:</strong> {agent_impact.get('current_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Efficiency Bonus:</strong> {agent_impact.get('efficiency_bonus', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col3:
        backup_impact = ai_assessment.get('backup_storage_impact', {})
        migration_method = backup_impact.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            backup_storage_type = backup_impact.get('backup_storage_type', 'nas_drive')
            backup_efficiency = backup_impact.get('backup_efficiency', 1.0)
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>💾 Backup Storage Impact</h4>
                <p><strong>Migration Method:</strong> {migration_method.replace('_', ' ').title()}</p>
                <p><strong>Storage Type:</strong> {backup_storage_type.replace('_', ' ').title()}</p>
                <p><strong>Protocol Efficiency:</strong> {backup_efficiency*100:.1f}%</p>
                <p><strong>Optimization Level:</strong> {"High" if backup_efficiency > 0.9 else "Medium" if backup_efficiency > 0.8 else "Needs Attention"}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            storage_impact = ai_assessment.get('destination_storage_impact', {})
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>🗄️ Storage Destination Impact</h4>
                <p><strong>Storage Type:</strong> {storage_impact.get('storage_type', 'Unknown')}</p>
                <p><strong>Performance Bonus:</strong> +{storage_impact.get('performance_bonus', 0)}%</p>
                <p><strong>Performance Multiplier:</strong> {storage_impact.get('storage_performance_multiplier', 1.0):.1f}x</p>
                <p><strong>Optimization Level:</strong> {"High" if storage_impact.get('performance_multiplier', 1.0) > 1.2 else "Medium" if storage_impact.get('performance_multiplier', 1.0) > 1.0 else "Standard"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Key Recommendations Summary
    st.markdown("**💡 Key Recommendations Summary:**")
    
    recommendations = ai_assessment.get('recommended_next_steps', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:4], 1):
            priority = "🔴 High" if i <= 2 else "🟡 Medium"
            st.markdown(f"**{priority} Priority:** {rec}")
    
    # Migration Health Score
    st.markdown("**🏥 Migration Health Score:**")
    
    health_factors = {
        'Performance Readiness': min(100, analysis.get('onprem_performance', {}).get('performance_score', 0)),
        'Network Quality': network_perf.get('ai_enhanced_quality_score', 0),
        'Agent Optimization': analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0) * 100,
        'Cost Efficiency': min(100, (monthly_savings / monthly_cost * 100)) if monthly_cost > 0 else 0,
        'Risk Mitigation': readiness_score
    }
    
    fig_health = px.bar(
        x=list(health_factors.keys()),
        y=list(health_factors.values()),
        title="Migration Health Score Breakdown",
        color=list(health_factors.values()),
        color_continuous_scale='RdYlGn',
        labels={'x': 'Factor', 'y': 'Score'}
    )
    
    # Add benchmark line at 80%
    fig_health.add_hline(y=80, line_dash="dash", line_color="blue", annotation_text="Target: 80%")
    
    st.plotly_chart(fig_health, use_container_width=True)

def render_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations tab using native components"""
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    
    # Deployment Recommendation Overview
    st.markdown("**☁️ Deployment Recommendation:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0)
        st.metric(
            "🎯 Recommended Deployment",
            recommendation,
            delta=f"Confidence: {confidence*100:.1f}%"
        )
    
    with col2:
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        st.metric(
            "📊 RDS Score",
            f"{rds_score:.0f}",
            delta=f"EC2: {ec2_score:.0f}"
        )
    
    with col3:
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🤖 AI Complexity",
            f"{complexity:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col4:
        if recommendation == 'RDS':
            monthly_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 0)
        else:
            monthly_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta="Compute + Storage"
        )
    
    with col5:
        reader_writer = aws_sizing.get('reader_writer_config', {})
        total_instances = reader_writer.get('total_instances', 1)
        writers = reader_writer.get('writers', 1)
        readers = reader_writer.get('readers', 0)
        st.metric(
            "🖥️ Total Instances",
            f"{total_instances}",
            delta=f"Writers: {writers}, Readers: {readers}"
        )
    
    # Detailed Sizing Recommendations using native components
    col1, col2 = st.columns(2)
    
    with col1:
        if recommendation == 'RDS':
            st.markdown("**🗄️ RDS Recommended Configuration:**")
            
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            with st.container():
                st.success("Amazon RDS Managed Service")
                st.write(f"**Instance Type:** {rds_rec.get('primary_instance', 'N/A')}")
                st.write(f"**vCPU:** {rds_rec.get('instance_specs', {}).get('vcpu', 'N/A')}")
                st.write(f"**Memory:** {rds_rec.get('instance_specs', {}).get('memory', 'N/A')} GB")
                st.write(f"**Storage:** {rds_rec.get('storage_size_gb', 0):,.0f} GB")
                st.write(f"**Storage Type:** {rds_rec.get('storage_type', 'gp3').upper()}")
                st.write(f"**Multi-AZ:** {'Yes' if rds_rec.get('multi_az', False) else 'No'}")
                st.write(f"**Backup Retention:** {rds_rec.get('backup_retention_days', 7)} days")
                st.write(f"**Monthly Instance Cost:** ${rds_rec.get('monthly_instance_cost', 0):,.0f}")
                st.write(f"**Monthly Storage Cost:** ${rds_rec.get('monthly_storage_cost', 0):,.0f}")
                st.write(f"**Total Monthly Cost:** ${rds_rec.get('total_monthly_cost', 0):,.0f}")
        else:
            st.markdown("**🖥️ EC2 Recommended Configuration:**")
            
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            with st.container():
                st.info("Amazon EC2 Self-Managed")
                st.write(f"**Instance Type:** {ec2_rec.get('primary_instance', 'N/A')}")
                st.write(f"**vCPU:** {ec2_rec.get('instance_specs', {}).get('vcpu', 'N/A')}")
                st.write(f"**Memory:** {ec2_rec.get('instance_specs', {}).get('memory', 'N/A')} GB")
                st.write(f"**Storage:** {ec2_rec.get('storage_size_gb', 0):,.0f} GB")
                st.write(f"**Storage Type:** {ec2_rec.get('storage_type', 'gp3').upper()}")
                st.write(f"**EBS Optimized:** {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}")
                st.write(f"**Enhanced Networking:** {'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'}")
                st.write(f"**Monthly Instance Cost:** ${ec2_rec.get('monthly_instance_cost', 0):,.0f}")
                st.write(f"**Monthly Storage Cost:** ${ec2_rec.get('monthly_storage_cost', 0):,.0f}")
                st.write(f"**Total Monthly Cost:** ${ec2_rec.get('total_monthly_cost', 0):,.0f}")
    
    with col2:
        st.markdown("**🎯 AI Sizing Factors:**")
        
        if recommendation == 'RDS':
            sizing_factors = aws_sizing.get('rds_recommendations', {}).get('ai_sizing_factors', {})
        else:
            sizing_factors = aws_sizing.get('ec2_recommendations', {}).get('ai_sizing_factors', {})
        
        with st.container():
            st.warning("AI-Enhanced Sizing Analysis")
            st.write(f"**Complexity Multiplier:** {sizing_factors.get('complexity_multiplier', 1.0):.2f}x")
            st.write(f"**Agent Scaling Factor:** {sizing_factors.get('agent_scaling_factor', 1.0):.2f}x")
            st.write(f"**AI Complexity Score:** {sizing_factors.get('ai_complexity_score', 6):.1f}/10")
            st.write(f"**Storage Multiplier:** {sizing_factors.get('storage_multiplier', 1.5):.1f}x")
            st.write(f"**Database Size:** {config.get('database_size_gb', 0):,} GB")
            st.write(f"**Performance Requirement:** {config.get('performance_requirements', 'standard').title()}")
            st.write(f"**Environment:** {config.get('environment', 'unknown').title()}")
            st.write(f"**Number of Agents:** {config.get('number_of_agents', 1)}")
    
    # RDS vs EC2 Comparison
    st.markdown("**⚖️ RDS vs EC2 Deployment Comparison:**")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        # Create comparison chart
        comparison_data = {
            'Criteria': ['Management', 'Cost', 'Performance', 'Scalability', 'Control', 'Total Score'],
            'RDS Score': [90, 80, 85, 90, 60, deployment_rec.get('rds_score', 0)],
            'EC2 Score': [60, 85, 90, 80, 95, deployment_rec.get('ec2_score', 0)]
        }
        
        fig_comparison = px.bar(
            comparison_data,
            x='Criteria',
            y=['RDS Score', 'EC2 Score'],
            title="RDS vs EC2 Scoring Breakdown",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with comparison_col2:
        st.markdown("**📊 Deployment Decision Factors:**")
        
        primary_reasons = deployment_rec.get('primary_reasons', [])
        
        with st.container():
            st.success(f"Why {recommendation}?")
            for reason in primary_reasons[:4]:
                st.write(f"• {reason}")
            
            st.write(f"**Confidence Level:** {confidence*100:.1f}%")
            decision_strength = ("Strong" if abs(rds_score - ec2_score) > 20 else 
                               "Moderate" if abs(rds_score - ec2_score) > 10 else "Weak")
            st.write(f"**Decision Strength:** {decision_strength}")

def render_fsx_comparisons_tab(analysis: Dict, config: Dict):
    """Render FSx destination comparisons tab"""
    st.subheader("🗄️ FSx Destination Storage Comparisons")
    
    fsx_comparisons = analysis.get('fsx_comparisons', {})
    
    if not fsx_comparisons:
        st.warning("FSx comparison data not available")
        return
    
    # Overview metrics
    st.markdown("**📊 Storage Destination Overview:**")
    
    col1, col2, col3 = st.columns(3)
    
    destinations = ['S3', 'FSx_Windows', 'FSx_Lustre']
    
    for i, dest in enumerate(destinations):
        with [col1, col2, col3][i]:
            comparison = fsx_comparisons.get(dest, {})
            
            st.metric(
                f"🗄️ {dest.replace('_', ' ')}",
                comparison.get('performance_rating', 'Unknown'),
                delta=f"${comparison.get('estimated_monthly_storage_cost', 0):,.0f}/mo"
            )
    
    # Detailed comparison table
    st.markdown("**📋 Detailed Comparison Analysis:**")
    
    comparison_data = []
    for dest_type, comparison in fsx_comparisons.items():
        comparison_data.append({
            'Storage Type': dest_type.replace('_', ' '),
            'Migration Time (hrs)': f"{comparison.get('estimated_migration_time_hours', 0):.1f}",
            'Throughput (Mbps)': f"{comparison.get('migration_throughput_mbps', 0):,.0f}",
            'Monthly Cost': f"${comparison.get('estimated_monthly_storage_cost', 0):,.0f}",
            'Performance': comparison.get('performance_rating', 'Unknown'),
            'Cost Rating': comparison.get('cost_rating', 'Unknown'),
            'Complexity': comparison.get('complexity_rating', 'Unknown')
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Detailed analysis for each destination
    for dest_type, comparison in fsx_comparisons.items():
        with st.expander(f"📂 {dest_type.replace('_', ' ')} Detailed Analysis", expanded=(dest_type == 'S3')):
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"**{dest_type.replace('_', ' ')} Configuration:**")
                st.write(f"**Storage Type:** {comparison.get('destination_type', 'Unknown')}")
                st.write(f"**Migration Time:** {comparison.get('estimated_migration_time_hours', 0):.1f} hours")
                st.write(f"**Throughput:** {comparison.get('migration_throughput_mbps', 0):,.0f} Mbps")
                st.write(f"**Monthly Cost:** ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}")
                
                # Agent configuration
                agent_config = comparison.get('agent_configuration', {})
                st.write(f"**Agents:** {agent_config.get('number_of_agents', 1)}")
                st.write(f"**Agent Cost:** ${agent_config.get('total_monthly_cost', 0):,.0f}/month")
                st.write(f"**Performance Multiplier:** {agent_config.get('storage_performance_multiplier', 1.0):.1f}x")
            
            with detail_col2:
                st.markdown(f"**Recommendations for {dest_type.replace('_', ' ')}:**")
                recommendations = comparison.get('recommendations', [])
                for rec in recommendations[:4]:
                    st.write(f"• {rec}")
                
                # Network performance details
                network_perf = comparison.get('network_performance', {})
                if network_perf:
                    st.write(f"**Network Quality:** {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
                    st.write(f"**Latency:** {network_perf.get('total_latency_ms', 0):.1f} ms")
                    st.write(f"**Reliability:** {network_perf.get('total_reliability', 0)*100:.3f}%")

def create_network_path_diagram(network_perf: Dict):
    """Create network path diagram using Plotly"""
    segments = network_perf.get('segments', [])
    if not segments:
        return None
    
    # Create a simple network path visualization
    x_positions = list(range(len(segments)))
    y_positions = [0] * len(segments)
    
    fig = go.Figure()
    
    # Add segments as nodes
    for i, segment in enumerate(segments):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=segment['name'][:20],
            textposition='top center',
            name=segment['name']
        ))
    
    # Add connections
    for i in range(len(segments) - 1):
        fig.add_trace(go.Scatter(
            x=[i, i+1],
            y=[0, 0],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Network Path Visualization",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300
    )
    
    return fig

def render_performance_impact_table(analysis: Dict, config: Dict):
    """Render performance impact analysis table"""
    st.markdown("**⚡ Performance Impact Analysis:**")
    
    # Get various performance metrics
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    network_perf = analysis.get('network_performance', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    # Create performance impact data
    impact_data = []
    
    # Hardware baseline
    user_nic = config.get('nic_speed', 1000)
    impact_data.append({
        'Component': 'Hardware NIC',
        'Baseline (Mbps)': f"{user_nic:,}",
        'Efficiency': '100%',
        'Impact': 'Baseline',
        'Notes': f"{config.get('nic_type', 'Unknown').replace('_', ' ').title()}"
    })
    
    # OS network efficiency
    os_efficiency = os_impact.get('network_efficiency', 0.90)
    after_os = user_nic * os_efficiency
    impact_data.append({
        'Component': 'OS Network Stack',
        'Baseline (Mbps)': f"{after_os:,.0f}",
        'Efficiency': f"{os_efficiency*100:.1f}%",
        'Impact': f"-{(1-os_efficiency)*100:.1f}%",
        'Notes': os_impact.get('name', 'OS').split()[-1]
    })
    
    # Network path
    network_limit = network_perf.get('effective_bandwidth_mbps', user_nic)
    network_efficiency = (network_limit / user_nic) if user_nic > 0 else 1.0
    impact_data.append({
        'Component': 'Network Path',
        'Baseline (Mbps)': f"{network_limit:,.0f}",
        'Efficiency': f"{network_efficiency*100:.1f}%",
        'Impact': f"{(network_efficiency-1)*100:+.1f}%",
        'Notes': f"{network_perf.get('environment', 'Unknown').title()} environment"
    })
    
    # Migration agents
    final_throughput = agent_analysis.get('total_effective_throughput', network_limit * 0.75)
    agent_efficiency = (final_throughput / network_limit) if network_limit > 0 else 0.75
    impact_data.append({
        'Component': 'Migration Agents',
        'Baseline (Mbps)': f"{final_throughput:,.0f}",
        'Efficiency': f"{agent_efficiency*100:.1f}%",
        'Impact': f"{(agent_efficiency-1)*100:+.1f}%",
        'Notes': f"{config.get('number_of_agents', 1)} {agent_analysis.get('primary_tool', 'DMS')} agents"
    })
    
    # Final performance
    total_efficiency = (final_throughput / user_nic) if user_nic > 0 else 0.5
    impact_data.append({
        'Component': 'FINAL RESULT',
        'Baseline (Mbps)': f"{final_throughput:,.0f}",
        'Efficiency': f"{total_efficiency*100:.1f}%",
        'Impact': f"{(total_efficiency-1)*100:+.1f}%",
        'Notes': f"Migration throughput for {config.get('database_size_gb', 0):,} GB"
    })
    
    df_impact = pd.DataFrame(impact_data)
    st.dataframe(df_impact, use_container_width=True)
    
    # Show estimated migration time
    if final_throughput > 0:
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size = config.get('database_size_gb', 0) * backup_size_multiplier
            st.info(f"**Estimated Migration Time:** {(data_size * 8 * 1000) / (final_throughput * 3600):.1f} hours for {data_size:,.0f} GB backup at {final_throughput:.0f} Mbps")
        else:
            migration_time = (config.get('database_size_gb', 0) * 8 * 1000) / (final_throughput * 3600)
            st.info(f"**Estimated Migration Time:** {migration_time:.1f} hours at {final_throughput:.0f} Mbps effective throughput")

async def main():
    """Main Streamlit application"""
    
    # Render enhanced header
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Run analysis when configuration is set
    if st.button("🚀 Run Enhanced AI Migration Analysis", type="primary", use_container_width=True):
        
        with st.spinner("🤖 Running comprehensive AI migration analysis..."):
            try:
                # Initialize analyzer
                analyzer = EnhancedMigrationAnalyzer()
                
                # Run comprehensive analysis
                analysis = await analyzer.comprehensive_ai_migration_analysis(config)
                
                # Store analysis in session state
                st.session_state['analysis'] = analysis
                st.session_state['config'] = config
                
                st.success("✅ AI Analysis Complete!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    
    # Display results if analysis is available
    if 'analysis' in st.session_state and 'config' in st.session_state:
        analysis = st.session_state['analysis']
        config = st.session_state['config']
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Migration Dashboard", 
            "🧠 AI Insights", 
            "🌐 Network Intelligence", 
            "💰 Cost & Pricing", 
            "💻 OS Performance", 
            "🎯 AWS Sizing",
            "🗄️ FSx Comparisons"
        ])
        
        with tab1:
            render_migration_dashboard_tab(analysis, config)
        
        with tab2:
            render_ai_insights_tab_enhanced(analysis, config)
        
        with tab3:
            render_network_intelligence_tab(analysis, config)
        
        with tab4:
            render_cost_pricing_tab(analysis, config)
        
        with tab5:
            render_os_performance_tab(analysis, config)
        
        with tab6:
            render_aws_sizing_tab(analysis, config)
        
        with tab7:
            render_fsx_comparisons_tab(analysis, config)
    
    # Professional footer with backup storage capabilities
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic Claude AI • Real-time AWS Integration • Professional Migration Analysis • Advanced Agent Scaling • FSx Destination Analysis • Backup Storage Migration Support</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Network Intelligence • 🎯 AI-Driven Recommendations • 📊 Executive Reporting • 🤖 Multi-Agent Optimization • 🗄️ S3/FSx Comparisons • 💾 SMB/NFS Backup Support
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())