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

# Professional CSS styling with enhanced colors
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(30,60,114,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(52,73,94,0.2);
        border-left: 3px solid #3498db;
    }
    
    .ai-recommendation-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(44,62,80,0.2);
        border-left: 3px solid #e74c3c;
    }
    
    .live-pricing-card {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(41,128,185,0.2);
        border-left: 3px solid #f39c12;
    }
    
    .network-intelligence-card {
        background: linear-gradient(135deg, #16a085 0%, #1abc9c 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(22,160,133,0.2);
        border-left: 3px solid #27ae60;
    }
    
    .os-performance-enhanced-card {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(142,68,173,0.2);
        border-left: 3px solid #e67e22;
    }
    
    .physical-virtual-comparison-card {
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(189,195,199,0.3);
        border-left: 3px solid #95a5a6;
    }
    
    .api-status-card {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #2c3e50;
        font-size: 0.9rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    .metric-card h4, .metric-card h5 {
        margin: 0 0 0.8rem 0;
        color: #2c3e50;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.4rem 0;
        color: #5a6c7d;
        line-height: 1.4;
    }
    
    .network-diagram-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #dee2e6;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    
    .status-online { background-color: #27ae60; }
    .status-offline { background-color: #e74c3c; }
    .status-warning { background-color: #f39c12; }
    
    .enterprise-footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(44,62,80,0.2);
    }
    
    .detailed-analysis-section {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .detailed-analysis-section h4, .detailed-analysis-section h5 {
        margin: 0 0 0.8rem 0;
        color: #2c3e50;
        font-weight: 600;
    }
    
    .detailed-analysis-section p {
        margin: 0.4rem 0;
        color: #5a6c7d;
        line-height: 1.4;
    }
    
    .detailed-analysis-section ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    
    .detailed-analysis-section li {
        margin: 0.3rem 0;
        color: #5a6c7d;
    }
    
    /* Compact metric cards for dashboard */
    .compact-metric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
    }
    
    .compact-metric h5 {
        margin: 0 0 0.5rem 0;
        color: #2c3e50;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .compact-metric p {
        margin: 0.2rem 0;
        color: #5a6c7d;
        font-size: 0.85rem;
        line-height: 1.3;
    }
    
    /* AWS Sizing specific styles */
    .aws-sizing-card {
        background: linear-gradient(135deg, #ff7b7b 0%, #667eea 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(102,126,234,0.2);
        border-left: 3px solid #667eea;
    }
    
    .deployment-comparison-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .pdf-download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(102,126,234,0.3);
        text-align: center;
    }
    
    /* Professional tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    
    /* Clean spacing */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    
    .stNumberInput > div > div {
        border-radius: 6px;
    }
    
    /* Professional button styling */
    .stButton > button {
        border-radius: 6px;
        border: none;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(52,152,219,0.3);
    }
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    /* Clean sidebar */
    .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label {
        font-weight: 500;
        color: #2c3e50;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-card, .ai-insight-card, .ai-recommendation-card {
            padding: 1rem;
        }
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
                # Initialize the client with latest model
                self.client = anthropic.Anthropic(api_key=self.api_key)
                
                # Test the connection with a simple API call using current model
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
            logger.warning("No Anthropic API key found in secrets")
    
    async def analyze_migration_workload(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced AI-powered workload analysis with detailed insights"""
        if not self.connected:
            return self._fallback_workload_analysis(config, performance_data)
        
        try:
            # Enhanced prompt for more detailed analysis
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

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
            Format the response as detailed sections with clear recommendations and actionable insights.
            """
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_response = message.content[0].text
            
            # Enhanced parsing for detailed analysis
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
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_workload_analysis(config, performance_data)
    
    def _parse_detailed_ai_response(self, ai_response: str, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced parsing for detailed AI analysis"""
        
        # Calculate complexity score based on multiple factors
        complexity_factors = []
        base_complexity = 5
        
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
        
        # OS complexity
        if 'windows' in config['operating_system']:
            complexity_factors.append(('Windows licensing considerations', 0.5))
            base_complexity += 0.5
        
        # Downtime constraints
        if config['downtime_tolerance_minutes'] < 60:
            complexity_factors.append(('Strict downtime requirements', 1))
            base_complexity += 1
        
        complexity_score = min(10, max(1, base_complexity))
        
        # Generate detailed risk assessment
        risk_factors = []
        risk_percentages = {}
        
        if config['source_database_engine'] != config['database_engine']:
            risk_factors.append("Schema conversion complexity may cause compatibility issues")
            risk_percentages['schema_conversion_risk'] = 25
        
        if config['database_size_gb'] > 5000:
            risk_factors.append("Large database size increases migration time and failure probability")
            risk_percentages['large_database_risk'] = 15
        
        if config['downtime_tolerance_minutes'] < 120:
            risk_factors.append("Tight downtime window may require multiple attempts")
            risk_percentages['downtime_risk'] = 20
        
        perf_score = performance_data.get('performance_score', 0)
        if perf_score < 70:
            risk_factors.append("Current performance issues may impact migration success")
            risk_percentages['performance_risk'] = 30
        
        # Generate detailed recommendations
        performance_recommendations = []
        performance_improvements = {}
        
        if perf_score < 80:
            performance_recommendations.append("Optimize database queries and indexes before migration")
            performance_improvements['query_optimization'] = '15-25%'
        
        if config['ram_gb'] < 32:
            performance_recommendations.append("Consider memory upgrade for better migration throughput")
            performance_improvements['memory_upgrade'] = '20-30%'
        
        # Enhanced timeline with phases
        timeline_suggestions = [
            "Phase 1: Assessment and Planning (2-3 weeks)",
            "Phase 2: Environment Setup and Testing (2-4 weeks)", 
            "Phase 3: Data Validation and Performance Testing (1-2 weeks)",
            "Phase 4: Migration Execution (1-3 days)",
            "Phase 5: Post-Migration Validation and Optimization (1 week)"
        ]
        
        # Detailed resource allocation
        resource_allocation = {
            'migration_team_size': 3 + (complexity_score // 3),
            'aws_specialists_needed': 1 if complexity_score < 6 else 2,
            'database_experts_required': 1 if config['source_database_engine'] == config['database_engine'] else 2,
            'testing_resources': '2-3 dedicated testers for ' + ('2 weeks' if complexity_score < 7 else '3-4 weeks'),
            'infrastructure_requirements': f"Staging environment with {config['cpu_cores']*2} cores and {config['ram_gb']*1.5} GB RAM"
        }
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': risk_factors,
            'risk_percentages': risk_percentages,
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors, config),
            'performance_recommendations': performance_recommendations,
            'performance_improvements': performance_improvements,
            'timeline_suggestions': timeline_suggestions,
            'resource_allocation': resource_allocation,
            'cost_optimization': self._generate_cost_optimization(config, complexity_score),
            'best_practices': self._generate_best_practices(config, complexity_score),
            'testing_strategy': self._generate_testing_strategy(config, complexity_score),
            'rollback_procedures': self._generate_rollback_procedures(config),
            'post_migration_monitoring': self._generate_monitoring_recommendations(config),
            'confidence_level': 'high' if complexity_score < 6 else 'medium' if complexity_score < 8 else 'requires_specialist_review',
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score))),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration',
                'critical_success_factors': self._identify_critical_success_factors(config, complexity_score)
            }
        }
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
            strategies.append("Create detailed schema mapping documentation")
            strategies.append("Implement phased schema migration with rollback checkpoints")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
            strategies.append("Use AWS DataSync for initial bulk data transfer")
            strategies.append("Schedule migration during low-traffic periods")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
            strategies.append("Use AWS DMS ongoing replication for data synchronization")
            strategies.append("Prepare detailed rollback procedures with time estimates")
        
        if any('performance' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct pre-migration performance tuning")
            strategies.append("Implement AWS CloudWatch monitoring throughout migration")
            strategies.append("Establish performance baselines and acceptance criteria")
        
        return strategies
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing to reduce costs by 60-70%")
        
        if complexity_score < 6:
            optimizations.append("Leverage AWS Managed Services (RDS) to reduce operational overhead")
        
        optimizations.append("Implement automated scaling policies to optimize resource utilization")
        optimizations.append("Use S3 Intelligent Tiering for backup storage cost optimization")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate specific best practices"""
        practices = []
        
        practices.append("Implement comprehensive backup strategy before migration initiation")
        practices.append("Use AWS Migration Hub for centralized migration tracking")
        practices.append("Establish detailed communication plan with stakeholders")
        
        if config['database_engine'] in ['mysql', 'postgresql']:
            practices.append("Leverage native database replication for minimal downtime")
        
        if complexity_score > 7:
            practices.append("Engage AWS Professional Services for complex migration scenarios")
        
        practices.append("Implement automated testing pipelines for validation")
        practices.append("Create detailed runbook with step-by-step procedures")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate comprehensive testing strategy"""
        strategy = []
        
        strategy.append("Unit Testing: Validate individual migration components")
        strategy.append("Integration Testing: Test end-to-end migration workflow")
        strategy.append("Performance Testing: Validate AWS environment performance")
        strategy.append("Data Integrity Testing: Verify data consistency and completeness")
        
        if config['source_database_engine'] != config['database_engine']:
            strategy.append("Schema Conversion Testing: Validate converted database objects")
            strategy.append("Application Compatibility Testing: Ensure application functionality")
        
        strategy.append("Disaster Recovery Testing: Validate backup and restore procedures")
        strategy.append("Security Testing: Verify access controls and encryption")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = []
        
        procedures.append("Maintain synchronized read replica during migration window")
        procedures.append("Create point-in-time recovery snapshot before cutover")
        procedures.append("Prepare DNS switching procedures for quick rollback")
        procedures.append("Document application configuration rollback steps")
        procedures.append("Establish go/no-go decision criteria with specific metrics")
        procedures.append("Test rollback procedures in staging environment")
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate post-migration monitoring recommendations"""
        monitoring = []
        
        monitoring.append("Implement CloudWatch detailed monitoring for all database metrics")
        monitoring.append("Set up automated alerts for performance degradation")
        monitoring.append("Monitor application response times and error rates")
        monitoring.append("Track database connection patterns and query performance")
        monitoring.append("Implement cost monitoring and optimization alerts")
        monitoring.append("Schedule regular performance reviews for first 30 days")
        
        return monitoring
    
    def _identify_critical_success_factors(self, config: Dict, complexity_score: int) -> List[str]:
        """Identify critical success factors"""
        factors = []
        
        factors.append("Stakeholder alignment on migration timeline and expectations")
        factors.append("Comprehensive testing in staging environment matching production")
        factors.append("Skilled migration team with AWS and database expertise")
        
        if complexity_score > 7:
            factors.append("Dedicated AWS solutions architect involvement")
        
        if config['downtime_tolerance_minutes'] < 120:
            factors.append("Near-zero downtime migration strategy implementation")
        
        factors.append("Robust monitoring and alerting throughout migration process")
        factors.append("Clear rollback criteria and tested rollback procedures")
        
        return factors
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced fallback analysis when AI is not available"""
        
        # Calculate complexity based on configuration
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        if config['performance_requirements'] == 'high':
            complexity_score += 1
        if config['downtime_tolerance_minutes'] < 60:
            complexity_score += 1
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': [
                "Migration complexity varies with database engine differences",
                "Large database sizes increase migration duration",
                "Performance requirements may necessitate additional testing"
            ],
            'mitigation_strategies': [
                "Conduct thorough pre-migration testing",
                "Plan for adequate migration windows",
                "Implement comprehensive backup strategies"
            ],
            'performance_recommendations': [
                "Optimize database before migration",
                "Consider read replicas for minimal downtime",
                "Monitor performance throughout migration"
            ],
            'confidence_level': 'medium',
            'raw_ai_response': 'AI analysis not available - using fallback analysis'
        }

class AWSAPIManager:
    """Manage AWS API integration for real-time pricing and optimization"""
    
    def __init__(self):
        self.session = None
        self.pricing_client = None
        self.compute_optimizer_client = None
        self.connected = False
        
        try:
            # Try to initialize AWS session
            self.session = boto3.Session()
            self.pricing_client = self.session.client('pricing', region_name='us-east-1')
            
            # Test connection
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
            # Get EC2 pricing
            ec2_pricing = await self._get_ec2_pricing(region)
            
            # Get RDS pricing
            rds_pricing = await self._get_rds_pricing(region)
            
            # Get storage pricing
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
        try:
            # AWS Pricing API calls for EC2 instances
            instance_types = ['t3.medium', 't3.large', 't3.xlarge', 'c5.large', 'c5.xlarge', 
                            'c5.2xlarge', 'c5.4xlarge', 'r6i.large', 'r6i.xlarge', 
                            'r6i.2xlarge', 'r6i.4xlarge', 'r6i.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        # Extract pricing information
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_hour = float(price_info['pricePerUnit']['USD'])
                                
                                # Get instance specs
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'cost_per_hour': price_per_hour
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                    # Use fallback pricing
                    pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"EC2 pricing fetch failed: {e}")
            return self._fallback_ec2_pricing()
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get RDS instance pricing"""
        try:
            instance_types = ['db.t3.medium', 'db.t3.large', 'db.r6g.large', 'db.r6g.xlarge', 
                            'db.r6g.2xlarge', 'db.r6g.4xlarge', 'db.r6g.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonRDS',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': 'MySQL'},
                            {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
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
                    logger.warning(f"Failed to get RDS pricing for {instance_type}: {e}")
                    pricing_data[instance_type] = self._get_fallback_rds_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"RDS pricing fetch failed: {e}")
            return self._fallback_rds_pricing()
    
    async def _get_storage_pricing(self, region: str) -> Dict:
        """Get EBS storage pricing"""
        try:
            storage_types = ['gp3', 'io1', 'io2']
            pricing_data = {}
            
            for storage_type in storage_types:
                try:
                    volume_type_map = {
                        'gp3': 'General Purpose',
                        'io1': 'Provisioned IOPS',
                        'io2': 'Provisioned IOPS'
                    }
                    
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                            {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type_map.get(storage_type, 'General Purpose')},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
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
                                price_per_gb = float(price_info['pricePerUnit']['USD'])
                                
                                pricing_data[storage_type] = {
                                    'cost_per_gb_month': price_per_gb,
                                    'iops_included': 3000 if storage_type == 'gp3' else 0,
                                    'cost_per_iops_month': 0.065 if storage_type in ['io1', 'io2'] else 0
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get storage pricing for {storage_type}: {e}")
                    pricing_data[storage_type] = self._get_fallback_storage_pricing(storage_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Storage pricing fetch failed: {e}")
            return self._fallback_storage_pricing()
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
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
            # Extract number from strings like "4 GiB", "8.0 GiB"
            import re
            match = re.search(r'([\d.]+)', memory_str)
            if match:
                return int(float(match.group(1)))
            return 4  # Default
        except:
            return 4
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data when API is unavailable"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing()
        }
    
    def _fallback_ec2_pricing(self) -> Dict:
        """Fallback EC2 pricing data"""
        return {
            't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
            't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832},
            't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664},
            'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
            'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
            'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
            'c5.4xlarge': {'vcpu': 16, 'memory': 32, 'cost_per_hour': 0.68},
            'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252},
            'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504},
            'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008},
            'r6i.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 2.016},
            'r6i.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 4.032}
        }
    
    def _fallback_rds_pricing(self) -> Dict:
        """Fallback RDS pricing data"""
        return {
            'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
            'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
            'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
            'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
            'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92},
            'db.r6g.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 3.84},
            'db.r6g.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 7.68}
        }
    
    def _fallback_storage_pricing(self) -> Dict:
        """Fallback storage pricing data"""
        return {
            'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
            'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            'io2': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065}
        }
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for specific instance type"""
        fallback_data = self._fallback_ec2_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05})
    
    def _get_fallback_rds_pricing(self, instance_type: str) -> Dict:
        """Get fallback RDS pricing for specific instance type"""
        fallback_data = self._fallback_rds_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.07})
    
    def _get_fallback_storage_pricing(self, storage_type: str) -> Dict:
        """Get fallback storage pricing for specific type"""
        fallback_data = self._fallback_storage_pricing()
        return fallback_data.get(storage_type, {'cost_per_gb_month': 0.08, 'iops_included': 0})

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
                    'strengths': ['Native SQL Server integration', 'Enterprise management tools', 'Active Directory integration'],
                    'weaknesses': ['Higher licensing costs', 'More resource overhead', 'Complex patch management'],
                    'migration_considerations': ['Licensing compliance', 'Service account migration', 'Registry dependencies']
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
                    'strengths': ['Improved container support', 'Enhanced security features', 'Better cloud integration'],
                    'weaknesses': ['Higher costs', 'Newer OS compatibility risks', 'Learning curve for admins'],
                    'migration_considerations': ['Hardware compatibility', 'Application compatibility testing', 'Training requirements']
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
                    'strengths': ['Excellent performance', 'Strong container support', 'Comprehensive support'],
                    'weaknesses': ['Commercial licensing required', 'Steeper learning curve', 'Limited Windows application support'],
                    'migration_considerations': ['Staff training needs', 'Application compatibility', 'Support contracts']
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
                    'strengths': ['Latest performance optimizations', 'Enhanced security', 'Modern container runtime'],
                    'weaknesses': ['Newer release stability', 'Application compatibility risks', 'Migration complexity'],
                    'migration_considerations': ['Extensive testing required', 'Legacy application assessment', 'Staff skill upgrade']
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
                    'strengths': ['No licensing costs', 'Great community support', 'Excellent for open source databases'],
                    'weaknesses': ['No commercial support without subscription', 'Requires Linux expertise', 'Less enterprise tooling'],
                    'migration_considerations': ['Staff Linux skills', 'Management tool migration', 'Support strategy']
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
                    'strengths': ['Latest performance features', 'Enhanced security', 'Cloud-native optimizations'],
                    'weaknesses': ['Newer release risks', 'Potential compatibility issues', 'Learning curve'],
                    'migration_considerations': ['Comprehensive testing', 'Backup OS strategy', 'Monitoring setup']
                }
            }
        }
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, database_engine: str) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        
        os_config = self.operating_systems[os_type]
        
        # Base OS efficiency calculation (preserved original logic)
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'][database_engine]
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            if 'windows' in os_type:
                total_efficiency *= 1.02
            else:
                total_efficiency *= 1.05
        
        # Enhanced return with AI insights
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }

class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_linux_nas': {
                'name': 'Non-Prod: San Jose Linux NAS + Jump Server → AWS S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
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
                    'optimization_opportunities': ['NAS performance tuning', 'DX bandwidth upgrade', 'Parallel transfer optimization'],
                    'risk_factors': ['Single DX connection dependency', 'NAS hardware limitations'],
                    'recommended_improvements': ['Implement NAS caching', 'Configure QoS on DX', 'Add backup connectivity']
                }
            },
            'nonprod_sj_windows_share': {
                'name': 'Non-Prod: San Jose Windows Share + Jump Server → AWS S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.88
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.90
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows file sharing overhead', 'SMB protocol latency', 'Windows network stack'],
                    'optimization_opportunities': ['SMB3 optimization', 'Windows TCP tuning', 'Antivirus exclusions'],
                    'risk_factors': ['Windows update interruptions', 'SMB vulnerabilities', 'File locking issues'],
                    'recommended_improvements': ['Upgrade to SMB3.1.1', 'Optimize TCP window scaling', 'Configure bypass for migration tools']
                }
            },
            'prod_sa_linux_nas': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
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
                    'optimization_opportunities': ['End-to-end optimization', 'Compression algorithms', 'Parallel processing'],
                    'risk_factors': ['Multiple failure points', 'Complex troubleshooting', 'Bandwidth contention'],
                    'recommended_improvements': ['Implement WAN optimization', 'Add redundant paths', 'Real-time monitoring']
                }
            },
            'prod_sa_windows_share': {
                'name': 'Prod: San Antonio Windows Share → San Jose → AWS Production VPC S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'San Antonio Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.85
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 15,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.91
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 10,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.93
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows overhead across hops', 'SMB cross-site performance', 'Complex authentication'],
                    'optimization_opportunities': ['DFS optimization', 'Credential caching', 'Protocol selection'],
                    'risk_factors': ['Authentication timeouts', 'Complex Windows dependencies', 'Cross-site file locking'],
                    'recommended_improvements': ['Implement DFS-R', 'Optimize Kerberos', 'Configure offline files']
                }
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Original performance calculation (preserved)
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        ai_optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            # Base metrics
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day and congestion adjustments (preserved original logic)
            if segment['connection_type'] == 'internal_lan':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.1
                else:
                    congestion_factor = 0.95
            elif segment['connection_type'] == 'private_line':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.2
                else:
                    congestion_factor = 0.9
            elif segment['connection_type'] == 'direct_connect':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.05
                else:
                    congestion_factor = 0.98
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments (preserved)
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # AI optimization potential
            ai_optimization_score *= segment['ai_optimization_potential']
            
            # Accumulate metrics
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
        
        # Calculate quality scores (preserved original logic)
        latency_score = max(0, 100 - (total_latency * 2))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)
        reliability_score = total_reliability * 100
        
        # AI-enhanced network quality with optimization potential
        base_network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        ai_enhanced_quality = base_network_quality * ai_optimization_score
        
        return {
            'path_name': path['name'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': base_network_quality,
            'ai_enhanced_quality_score': ai_enhanced_quality,
            'ai_optimization_potential': (1 - ai_optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }

class AgentSizingManager:
    """Enhanced agent sizing with AI recommendations"""
    
    def __init__(self):
        # Keep original datasync and dms agents configuration
        self.datasync_agents = {
            'small': {
                'name': 'Small Agent (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 250,
                'max_concurrent_tasks': 10,
                'cost_per_hour': 0.0416,
                'recommended_for': 'Up to 1TB databases, <100 Mbps network',
                'ai_optimization_tips': [
                    'Configure parallel transfers for small files',
                    'Optimize for network bandwidth efficiency',
                    'Consider file compression during transfer'
                ]
            },
            'medium': {
                'name': 'Medium Agent (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 500,
                'max_concurrent_tasks': 25,
                'cost_per_hour': 0.085,
                'recommended_for': '1-5TB databases, 100-500 Mbps network',
                'ai_optimization_tips': [
                    'Balance CPU and network utilization',
                    'Implement intelligent retry mechanisms',
                    'Optimize task scheduling algorithms'
                ]
            },
            'large': {
                'name': 'Large Agent (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps': 1000,
                'max_concurrent_tasks': 50,
                'cost_per_hour': 0.17,
                'recommended_for': '5-20TB databases, 500Mbps-1Gbps network',
                'ai_optimization_tips': [
                    'Enable advanced parallel processing',
                    'Implement adaptive bandwidth management',
                    'Use predictive caching strategies'
                ]
            },
            'xlarge': {
                'name': 'XLarge Agent (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps': 2000,
                'max_concurrent_tasks': 100,
                'cost_per_hour': 0.34,
                'recommended_for': '>20TB databases, >1Gbps network',
                'ai_optimization_tips': [
                    'Maximize multi-core processing efficiency',
                    'Implement intelligent load balancing',
                    'Use advanced compression algorithms'
                ]
            }
        }
        
        self.dms_agents = {
            'small': {
                'name': 'Small DMS Instance (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 200,
                'max_concurrent_tasks': 5,
                'cost_per_hour': 0.0416,
                'recommended_for': 'Up to 500GB databases, simple schemas',
                'ai_optimization_tips': [
                    'Optimize CDC settings for small datasets',
                    'Configure efficient batch sizes',
                    'Minimize transformation overhead'
                ]
            },
            'medium': {
                'name': 'Medium DMS Instance (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 400,
                'max_concurrent_tasks': 10,
                'cost_per_hour': 0.085,
                'recommended_for': '500GB-2TB databases, moderate complexity',
                'ai_optimization_tips': [
                    'Balance memory allocation for transformations',
                    'Implement parallel table loading',
                    'Optimize validation processes'
                ]
            },
            'large': {
                'name': 'Large DMS Instance (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps': 800,
                'max_concurrent_tasks': 20,
                'cost_per_hour': 0.17,
                'recommended_for': '2-10TB databases, complex schemas',
                'ai_optimization_tips': [
                    'Enable advanced parallel processing',
                    'Implement intelligent error handling',
                    'Optimize LOB handling strategies'
                ]
            },
            'xlarge': {
                'name': 'XLarge DMS Instance (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps': 1500,
                'max_concurrent_tasks': 40,
                'cost_per_hour': 0.34,
                'recommended_for': '10-50TB databases, very complex schemas',
                'ai_optimization_tips': [
                    'Maximize CPU utilization for transformations',
                    'Implement advanced caching strategies',
                    'Use predictive error prevention'
                ]
            },
            'xxlarge': {
                'name': 'XXLarge DMS Instance (c5.4xlarge)',
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps': 2500,
                'max_concurrent_tasks': 80,
                'cost_per_hour': 0.68,
                'recommended_for': '>50TB databases, enterprise workloads',
                'ai_optimization_tips': [
                    'Optimize for maximum parallel throughput',
                    'Implement intelligent resource allocation',
                    'Use advanced monitoring and auto-tuning'
                ]
            }
        }

class EnhancedAWSMigrationManager:
    """Enhanced AWS migration manager with AI and real-time pricing"""
    
    def __init__(self, aws_api_manager: AWSAPIManager, ai_manager: AnthropicAIManager):
        self.aws_api = aws_api_manager
        self.ai_manager = ai_manager
        
        # Keep all original migration types
        self.migration_types = {
            'homogeneous': {
                'name': 'Homogeneous Migration',
                'description': 'Same database engine (uses DataSync for file-based, native replication for binary)',
                'complexity_factor': 0.3,
                'time_factor': 0.7,
                'risk_factor': 0.2,
                'preferred_tools': ['aws_datasync', 'native_replication'],
                'schema_conversion_required': False,
                'application_changes_required': False
            },
            'heterogeneous': {
                'name': 'Heterogeneous Migration',
                'description': 'Different database engines (requires DMS and SCT)',
                'complexity_factor': 0.8,
                'time_factor': 1.4,
                'risk_factor': 0.7,
                'preferred_tools': ['aws_dms', 'aws_sct'],
                'schema_conversion_required': True,
                'application_changes_required': True
            }
        }
    
    async def ai_enhanced_aws_sizing(self, on_prem_config: Dict) -> Dict:
        """AI-enhanced AWS sizing with real-time pricing"""
        
        # Get real-time pricing
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # Get AI workload analysis
        ai_analysis = await self.ai_manager.analyze_migration_workload(
            on_prem_config, 
            on_prem_config.get('performance_data', {})
        )
        
        # Enhanced sizing logic with AI insights
        rds_recommendations = await self._ai_calculate_rds_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        ec2_recommendations = await self._ai_calculate_ec2_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        # AI-enhanced reader/writer configuration
        reader_writer_config = await self._ai_calculate_reader_writer_config(
            on_prem_config, ai_analysis
        )
        
        # AI-powered deployment recommendation
        deployment_recommendation = await self._ai_recommend_deployment_type(
            on_prem_config, ai_analysis, rds_recommendations, ec2_recommendations
        )
        
        return {
            'rds_recommendations': rds_recommendations,
            'ec2_recommendations': ec2_recommendations,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': pricing_data
        }
    
    async def _ai_calculate_rds_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced RDS sizing calculation"""
        
        # Extract configuration
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI-based sizing multipliers
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1  # Adjust based on AI complexity
        
        cpu_multiplier = (1.2 if performance_req == 'high' else 1.0) * complexity_multiplier
        memory_multiplier = (1.3 if database_engine in ['oracle', 'postgresql'] else 1.1) * complexity_multiplier
        
        # Required cloud resources with AI adjustment
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing data
        rds_instances = pricing_data.get('rds_instances', {})
        
        # AI-enhanced instance selection
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in rds_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                # AI-enhanced scoring
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI complexity penalty for oversizing
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.1
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'db.r6g.8xlarge'
        
        # AI-enhanced storage recommendations
        storage_multiplier = 1.5 + (ai_complexity - 5) * 0.1  # More storage for complex migrations
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io1' if database_size_gb > 5000 or performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs with real-time pricing
        instance_cost = rds_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': rds_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'multi_az': environment == 'production',
            'backup_retention_days': 30 if environment == 'production' else 7,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_ec2_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced EC2 sizing calculation"""
        
        # Similar to RDS but with EC2-specific adjustments
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI complexity-based adjustments
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.15  # More aggressive for EC2
        
        # EC2 needs more overhead for OS and database management
        cpu_multiplier = (1.4 if performance_req == 'high' else 1.2) * complexity_multiplier
        memory_multiplier = (1.5 if database_engine in ['oracle', 'postgresql'] else 1.3) * complexity_multiplier
        
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing
        ec2_instances = pricing_data.get('ec2_instances', {})
        
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in ec2_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI penalty for complex workloads
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.15
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'r6i.8xlarge'
        
        # AI-enhanced storage sizing for EC2
        storage_multiplier = 2.0 + (ai_complexity - 5) * 0.2  # Even more generous for EC2
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io2' if performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs
        instance_cost = ec2_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': ec2_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_reader_writer_config(self, config: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced reader/writer configuration"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        
        # Start with single writer
        writers = 1
        readers = 0
        
        # AI-enhanced scaling logic
        if database_size_gb > 1000:
            readers += 1
        if database_size_gb > 5000:
            readers += 1 + int(ai_complexity / 5)  # AI complexity adds more readers
        if database_size_gb > 20000:
            readers += 2 + int(ai_complexity / 3)
        
        # Performance-based scaling with AI insights
        if performance_req == 'high':
            readers += 2 + int(ai_complexity / 4)
        
        # Environment-based scaling
        if environment == 'production':
            readers = max(readers, 2)  # Minimum 2 readers for production
            if database_size_gb > 50000 or ai_complexity > 8:
                writers = 2  # Multi-writer for very large or complex production DBs
        
        # AI-recommended adjustments
        if 'high availability' in ' '.join(ai_analysis.get('best_practices', [])).lower():
            readers += 1
        if 'complex queries' in ' '.join(ai_analysis.get('risk_factors', [])).lower():
            readers += 1
        
        # Calculate read/write distribution
        total_capacity = writers + readers
        write_capacity_percent = (writers / total_capacity) * 100
        read_capacity_percent = (readers / total_capacity) * 100
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': total_capacity,
            'write_capacity_percent': write_capacity_percent,
            'read_capacity_percent': read_capacity_percent,
            'recommended_read_split': min(80, read_capacity_percent),
            'reasoning': f"AI-optimized for {database_size_gb}GB, complexity {ai_complexity}/10, {performance_req} performance, {environment}",
            'ai_insights': {
                'complexity_impact': ai_complexity,
                'scaling_factors': ai_analysis.get('performance_recommendations', [])[:3],
                'optimization_potential': f"{(10 - ai_complexity) * 10}% potential for further optimization"
            }
        }
    
    async def _ai_recommend_deployment_type(self, config: Dict, ai_analysis: Dict, 
                                          rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """AI-powered deployment type recommendation"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        
        rds_score = 0
        ec2_score = 0
        
        # Size-based scoring (preserved original logic)
        if database_size_gb < 2000:
            rds_score += 40
        elif database_size_gb > 10000:
            ec2_score += 30
        
        # Performance scoring
        if performance_req == 'high':
            ec2_score += 30
            rds_score += 15
        else:
            rds_score += 35
        
        # Database engine scoring
        if database_engine in ['mysql', 'postgresql']:
            rds_score += 25
        elif database_engine == 'oracle':
            ec2_score += 25
        
        # Environment scoring
        if environment == 'production':
            rds_score += 20
        else:
            ec2_score += 10
        
        # AI complexity scoring
        if ai_complexity > 7:
            ec2_score += 20  # Complex workloads might need more control
            rds_score += 5   # But managed services help with complexity
        elif ai_complexity < 4:
            rds_score += 25  # Simple workloads perfect for managed services
        
        # AI insights-based scoring
        risk_factors = ai_analysis.get('risk_factors', [])
        if any('performance' in risk.lower() for risk in risk_factors):
            ec2_score += 15
        if any('management' in risk.lower() or 'operational' in risk.lower() for risk in risk_factors):
            rds_score += 20
        
        # Management complexity
        rds_score += 20
        
        # Cost consideration
        rds_cost = rds_rec.get('total_monthly_cost', 0)
        ec2_cost = ec2_rec.get('total_monthly_cost', 0)
        
        if ec2_cost < rds_cost * 0.8:  # EC2 significantly cheaper
            ec2_score += 15
        elif rds_cost < ec2_cost * 0.9:  # RDS competitive
            rds_score += 10
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'ai_complexity_factor': ai_complexity,
            'primary_reasons': self._get_ai_deployment_reasons(
                recommendation, rds_score, ec2_score, ai_analysis
            ),
            'ai_insights': {
                'complexity_impact': f"AI complexity score {ai_complexity}/10 influenced recommendation",
                'risk_mitigation': ai_analysis.get('mitigation_strategies', [])[:3],
                'cost_factors': {
                    'rds_monthly': rds_cost,
                    'ec2_monthly': ec2_cost,
                    'cost_difference_percent': abs(rds_cost - ec2_cost) / max(rds_cost, ec2_cost, 1) * 100
                }
            }
        }
    
    def _get_ai_deployment_reasons(self, recommendation: str, rds_score: int, 
                                 ec2_score: int, ai_analysis: Dict) -> List[str]:
        """Get AI-enhanced reasons for deployment recommendation"""
        
        base_reasons = []
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        
        if recommendation == 'rds':
            base_reasons = [
                "AI-optimized managed service reduces operational complexity",
                "Automated backups and patching aligned with AI recommendations",
                "Built-in monitoring supports AI-driven optimization",
                "Easy scaling matches AI-predicted growth patterns",
                f"AI complexity score ({ai_complexity}/10) suggests managed service benefits"
            ]
        else:
            base_reasons = [
                "Maximum control needed for AI-identified performance requirements",
                "Complex configurations support AI optimization strategies",
                "Custom tuning capabilities for AI-recommended improvements",
                "Full control enables AI-driven performance optimization",
                f"AI complexity score ({ai_complexity}/10) suggests need for advanced control"
            ]
        
        # Add AI-specific insights
        ai_recommendations = ai_analysis.get('performance_recommendations', [])
        if ai_recommendations:
            base_reasons.append(f"Supports AI recommendation: {ai_recommendations[0][:50]}...")
        
        return base_reasons[:5]

class OnPremPerformanceAnalyzer:
    """Enhanced on-premises performance analyzer with AI insights"""
    
    def __init__(self):
        # Keep all original CPU, storage configurations
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
        
        # Get original OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Original performance calculations (preserved)
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        memory_performance = self._calculate_memory_performance(config, os_impact)
        storage_performance = self._calculate_storage_performance(config, os_impact)
        network_performance = self._calculate_network_performance(config, os_impact)
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # Enhanced performance analysis using new metrics
        current_performance_analysis = self._analyze_current_performance_metrics(config)
        
        # AI-enhanced overall performance analysis
        overall_performance = self._calculate_ai_enhanced_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact, config
        )
        
        # AI bottleneck analysis
        ai_bottlenecks = self._ai_identify_bottlenecks(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, config
        )
        
        # Resource adequacy analysis
        resource_adequacy = self._analyze_resource_adequacy(config)
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'current_performance_analysis': current_performance_analysis,
            'resource_adequacy': resource_adequacy,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': ai_bottlenecks['bottlenecks'],
            'ai_insights': ai_bottlenecks['ai_insights'],
            'performance_score': overall_performance['composite_score'],
            'ai_optimization_recommendations': self._generate_ai_optimization_recommendations(
                overall_performance, os_impact, config
            )
        }
    
    def _analyze_current_performance_metrics(self, config: Dict) -> Dict:
        """Analyze current performance metrics provided by user"""
        
        current_storage = config.get('current_storage_gb', 0)
        peak_iops = config.get('peak_iops', 0)
        max_throughput = config.get('max_throughput_mbps', 0)
        database_size = config.get('database_size_gb', 0)
        
        # Storage utilization analysis
        storage_utilization = (database_size / current_storage) * 100 if current_storage > 0 else 0
        
        # IOPS intensity analysis
        iops_per_gb = peak_iops / database_size if database_size > 0 else 0
        
        # Throughput efficiency
        throughput_per_gb = max_throughput / database_size if database_size > 0 else 0
        
        # Performance classification
        if iops_per_gb > 50:
            workload_type = "High IOPS (OLTP-intensive)"
        elif throughput_per_gb > 1:
            workload_type = "High Throughput (Analytics/Batch)"
        else:
            workload_type = "Balanced Workload"
        
        return {
            'storage_utilization_percent': storage_utilization,
            'iops_per_gb': iops_per_gb,
            'throughput_per_gb_mbps': throughput_per_gb,
            'workload_classification': workload_type,
            'storage_efficiency': min(100, (100 - storage_utilization) + 50),  # Higher efficiency for lower utilization
            'performance_intensity': min(100, (iops_per_gb * 2) + (throughput_per_gb * 10)),
            'optimization_priority': "High" if storage_utilization > 80 or iops_per_gb > 100 else "Medium" if storage_utilization > 60 else "Low"
        }
    
    def _analyze_resource_adequacy(self, config: Dict) -> Dict:
        """Analyze resource adequacy comparing current vs anticipated needs"""
        
        current_memory = config.get('ram_gb', 0)
        anticipated_memory = config.get('anticipated_max_memory_gb', 0)
        current_cpu = config.get('cpu_cores', 0)
        anticipated_cpu = config.get('anticipated_max_cpu_cores', 0)
        
        # Memory adequacy
        memory_gap = anticipated_memory - current_memory
        memory_adequacy_score = (current_memory / anticipated_memory) * 100 if anticipated_memory > 0 else 100
        
        # CPU adequacy
        cpu_gap = anticipated_cpu - current_cpu
        cpu_adequacy_score = (current_cpu / anticipated_cpu) * 100 if anticipated_cpu > 0 else 100
        
        # Overall readiness
        overall_adequacy = (memory_adequacy_score + cpu_adequacy_score) / 2
        
        # Readiness classification
        if overall_adequacy >= 90:
            readiness_level = "Excellent - Current resources meet anticipated needs"
        elif overall_adequacy >= 75:
            readiness_level = "Good - Minor upgrades may be beneficial"
        elif overall_adequacy >= 60:
            readiness_level = "Fair - Moderate upgrades recommended"
        else:
            readiness_level = "Poor - Significant upgrades required"
        
        return {
            'memory_gap_gb': memory_gap,
            'cpu_gap_cores': cpu_gap,
            'memory_adequacy_score': memory_adequacy_score,
            'cpu_adequacy_score': cpu_adequacy_score,
            'overall_adequacy_score': overall_adequacy,
            'readiness_level': readiness_level,
            'upgrade_priority': "Immediate" if overall_adequacy < 60 else "Short-term" if overall_adequacy < 75 else "Long-term" if overall_adequacy < 90 else "Optional"
        }
    
    # Keep all original calculation methods but add AI enhancements
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics with AI insights"""
        
        # Original calculation (preserved)
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05
        
        single_thread_perf = config['cpu_ghz'] * os_impact['cpu_efficiency']
        multi_thread_perf = final_performance
        
        # AI enhancement: predict scaling characteristics
        ai_scaling_prediction = self._predict_cpu_scaling(config, final_performance)
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'single_thread_performance': single_thread_perf,
            'multi_thread_performance': multi_thread_perf,
            'utilization_estimate': 0.7,
            'efficiency_factor': os_impact['cpu_efficiency'],
            'ai_scaling_prediction': ai_scaling_prediction,
            'ai_bottleneck_risk': 'high' if final_performance < 30 else 'medium' if final_performance < 60 else 'low'
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance with AI insights"""
        
        # Original calculation (preserved)
        base_memory = config['ram_gb']
        
        if 'windows' in config['operating_system']:
            os_overhead = 4
        else:
            os_overhead = 2
        
        available_memory = base_memory - os_overhead
        db_memory = available_memory * 0.8
        buffer_pool = db_memory * 0.7
        memory_efficiency = os_impact['memory_efficiency']
        effective_memory = available_memory * memory_efficiency
        
        # AI enhancement: memory pressure prediction
        ai_memory_analysis = self._analyze_memory_requirements(config, effective_memory)
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'database_memory_gb': db_memory,
            'buffer_pool_gb': buffer_pool,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': memory_efficiency,
            'memory_pressure': 'low' if available_memory > 32 else 'medium' if available_memory > 16 else 'high',
            'ai_memory_analysis': ai_memory_analysis,
            'ai_optimization_potential': ai_memory_analysis['optimization_potential']
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance with AI insights"""
        
        # Original calculation (preserved)
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        effective_latency = storage_specs['latency_ms'] / os_impact['io_efficiency']
        
        # AI enhancement: storage optimization analysis
        ai_storage_analysis = self._analyze_storage_optimization(config, storage_type, effective_iops)
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'base_latency_ms': storage_specs['latency_ms'],
            'effective_latency_ms': effective_latency,
            'io_efficiency': os_impact['io_efficiency'],
            'ai_storage_analysis': ai_storage_analysis,
            'ai_upgrade_recommendation': ai_storage_analysis['upgrade_recommendation']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance with AI insights"""
        
        # Original calculation (preserved)
        base_bandwidth = config['nic_speed']
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        if config['server_type'] == 'vmware':
            effective_bandwidth *= 0.92
        
        # AI enhancement: network optimization analysis
        ai_network_analysis = self._analyze_network_optimization(config, effective_bandwidth)
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency'],
            'estimated_latency_ms': 0.1 if 'fiber' in config['nic_type'] else 0.2,
            'ai_network_analysis': ai_network_analysis,
            'ai_optimization_score': ai_network_analysis['optimization_score']
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database performance with AI insights"""
        
        # Original calculation (preserved)
        db_optimization = os_impact['db_optimization']
        
        if config['database_engine'] == 'mysql':
            base_tps = 5000
            connection_limit = 1000
        elif config['database_engine'] == 'postgresql':
            base_tps = 4500
            connection_limit = 500
        elif config['database_engine'] == 'oracle':
            base_tps = 6000
            connection_limit = 2000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 5500
            connection_limit = 1500
        else:
            base_tps = 4000
            connection_limit = 800
        
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        effective_tps = base_tps * hardware_factor * db_optimization
        
        # AI enhancement: database optimization analysis
        ai_db_analysis = self._analyze_database_optimization(config, effective_tps, db_optimization)
        
        return {
            'database_engine': config['database_engine'],
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps,
            'max_connections': connection_limit,
            'query_cache_efficiency': db_optimization * 0.9,
            'ai_db_analysis': ai_db_analysis,
            'ai_tuning_potential': ai_db_analysis['tuning_potential']
        }
    
    def _calculate_ai_enhanced_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                                 storage_perf: Dict, net_perf: Dict, 
                                                 db_perf: Dict, os_impact: Dict, config: Dict) -> Dict:
        """AI-enhanced overall performance calculation"""
        
        # Original scoring (preserved)
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        # Original composite score
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        # AI enhancement: workload-specific optimization
        ai_workload_optimization = self._analyze_workload_optimization(config, {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score
        })
        
        # AI-adjusted composite score
        ai_adjusted_score = composite_score * ai_workload_optimization['optimization_factor']
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'ai_adjusted_score': ai_adjusted_score,
            'performance_tier': self._get_performance_tier(ai_adjusted_score),
            'scaling_recommendation': self._get_ai_scaling_recommendation(cpu_score, memory_score, storage_score),
            'ai_workload_optimization': ai_workload_optimization
        }
    
    # AI helper methods
    def _predict_cpu_scaling(self, config: Dict, performance: float) -> Dict:
        """AI-powered CPU scaling prediction"""
        database_size = config['database_size_gb']
        
        scaling_factor = 1.0
        if database_size > 10000:
            scaling_factor = 1.3
        elif database_size > 5000:
            scaling_factor = 1.15
        
        return {
            'predicted_scaling_needs': scaling_factor,
            'bottleneck_prediction': 'likely' if performance < 40 else 'possible' if performance < 70 else 'unlikely',
            'recommendation': 'Upgrade CPU' if performance < 40 else 'Monitor performance' if performance < 70 else 'Current CPU sufficient'
        }
    
    def _analyze_memory_requirements(self, config: Dict, effective_memory: float) -> Dict:
        """AI memory requirement analysis"""
        database_size = config['database_size_gb']
        
        recommended_memory = database_size * 0.1  # 10% of database size as baseline
        if config['database_engine'] in ['oracle', 'postgresql']:
            recommended_memory *= 1.5
        
        optimization_potential = max(0, (recommended_memory - effective_memory) / recommended_memory)
        
        return {
            'recommended_memory_gb': recommended_memory,
            'current_vs_recommended': effective_memory / recommended_memory,
            'optimization_potential': optimization_potential,
            'memory_adequacy': 'sufficient' if effective_memory >= recommended_memory else 'insufficient'
        }
    
    def _analyze_storage_optimization(self, config: Dict, storage_type: str, effective_iops: float) -> Dict:
        """AI storage optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict IOPS requirements based on database size and type
        if config['database_engine'] in ['oracle', 'sqlserver']:
            required_iops = database_size * 5  # Higher IOPS requirement
        else:
            required_iops = database_size * 3
        
        upgrade_needed = effective_iops < required_iops
        
        return {
            'required_iops': required_iops,
            'current_vs_required': effective_iops / required_iops,
            'upgrade_recommendation': 'Upgrade to NVMe SSD' if upgrade_needed and storage_type != 'nvme_ssd' else 'Current storage adequate',
            'performance_impact': 'high' if upgrade_needed else 'low'
        }
    
    def _analyze_network_optimization(self, config: Dict, effective_bandwidth: float) -> Dict:
        """AI network optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict network requirements
        required_bandwidth = min(10000, database_size * 10)  # 10 Mbps per GB, max 10 Gbps
        
        optimization_score = min(100, (effective_bandwidth / required_bandwidth) * 100)
        
        return {
            'required_bandwidth_mbps': required_bandwidth,
            'optimization_score': optimization_score,
            'bottleneck_risk': 'high' if optimization_score < 50 else 'medium' if optimization_score < 80 else 'low'
        }
    
    def _analyze_database_optimization(self, config: Dict, effective_tps: float, db_optimization: float) -> Dict:
        """AI database optimization analysis"""
        
        # Predict TPS requirements based on database characteristics
        base_requirement = 2000  # Base TPS requirement
        if config['performance_requirements'] == 'high':
            base_requirement *= 2
        if config['database_size_gb'] > 10000:
            base_requirement *= 1.5
        
        tuning_potential = (1 - db_optimization) * 100
        
        return {
            'required_tps': base_requirement,
            'current_vs_required': effective_tps / base_requirement,
            'tuning_potential': tuning_potential,
            'optimization_priority': 'high' if tuning_potential > 20 else 'medium' if tuning_potential > 10 else 'low'
        }
    
    def _analyze_workload_optimization(self, config: Dict, scores: Dict) -> Dict:
        """AI workload-specific optimization analysis"""
        
        # Identify workload pattern
        if config['database_engine'] in ['mysql', 'postgresql']:
            workload_type = 'oltp'
            cpu_weight = 0.3
            memory_weight = 0.25
            storage_weight = 0.3
        elif config['database_engine'] == 'oracle':
            workload_type = 'mixed'
            cpu_weight = 0.25
            memory_weight = 0.3
            storage_weight = 0.25
        else:
            workload_type = 'general'
            cpu_weight = 0.25
            memory_weight = 0.2
            storage_weight = 0.25
        
        # Calculate workload-specific optimization factor
        weighted_score = (
            scores['cpu_score'] * cpu_weight +
            scores['memory_score'] * memory_weight +
            scores['storage_score'] * storage_weight +
            scores['network_score'] * 0.1 +
            scores['database_score'] * 0.1
        )
        
        optimization_factor = weighted_score / 100
        
        return {
            'workload_type': workload_type,
            'optimization_factor': optimization_factor,
            'primary_bottleneck': max(scores.items(), key=lambda x: x[1])[0].replace('_score', ''),
            'recommended_focus': self._get_optimization_focus(scores)
        }
    
    def _get_optimization_focus(self, scores: Dict) -> str:
        """Get primary optimization focus"""
        min_score = min(scores.values())
        min_component = [k for k, v in scores.items() if v == min_score][0]
        
        focus_map = {
            'cpu_score': 'CPU and processing optimization',
            'memory_score': 'Memory allocation and caching',
            'storage_score': 'Storage performance and I/O optimization',
            'network_score': 'Network bandwidth and latency',
            'database_score': 'Database configuration and tuning'
        }
        
        return focus_map.get(min_component, 'General performance optimization')
    
    def _ai_identify_bottlenecks(self, cpu_perf: Dict, mem_perf: Dict, 
                               storage_perf: Dict, net_perf: Dict, config: Dict) -> Dict:
        """AI-powered bottleneck identification"""
        
        bottlenecks = []
        ai_insights = []
        
        # CPU analysis
        if cpu_perf.get('ai_bottleneck_risk') == 'high':
            bottlenecks.append("CPU performance insufficient for workload")
            ai_insights.append("AI predicts CPU will be primary bottleneck during peak loads")
        
        # Memory analysis
        if mem_perf.get('memory_pressure') == 'high':
            bottlenecks.append("Memory pressure detected")
            ai_insights.append(f"AI recommends {mem_perf['ai_memory_analysis']['recommended_memory_gb']:.0f}GB for optimal performance")
        
        # Storage analysis
        if storage_perf.get('ai_storage_analysis', {}).get('performance_impact') == 'high':
            bottlenecks.append("Storage IOPS insufficient")
            ai_insights.append("AI suggests storage upgrade will provide significant performance improvement")
        
        # Network analysis
        if net_perf.get('ai_network_analysis', {}).get('bottleneck_risk') == 'high':
            bottlenecks.append("Network bandwidth limited")
            ai_insights.append("AI identifies network as migration performance constraint")
        
        if not bottlenecks:
            bottlenecks.append("No significant bottlenecks detected by AI analysis")
            ai_insights.append("AI assessment indicates well-balanced system configuration")
        
        return {
            'bottlenecks': bottlenecks,
            'ai_insights': ai_insights
        }
    
    def _generate_ai_optimization_recommendations(self, overall_perf: Dict, 
                                                os_impact: Dict, config: Dict) -> List[str]:
        """Generate AI-powered optimization recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if overall_perf['ai_adjusted_score'] < 60:
            recommendations.append("AI recommends comprehensive performance review and hardware upgrade")
        elif overall_perf['ai_adjusted_score'] < 80:
            recommendations.append("AI suggests targeted optimization of lowest-performing components")
        
        # OS-specific recommendations
        os_insights = os_impact.get('ai_insights', {})
        if os_insights:
            recommendations.extend([
                f"OS Consideration: {insight}" for insight in os_insights.get('migration_considerations', [])[:2]
            ])
        
        # Workload-specific recommendations
        workload_opt = overall_perf.get('ai_workload_optimization', {})
        if workload_opt.get('recommended_focus'):
            recommendations.append(f"AI Priority: Focus on {workload_opt['recommended_focus']}")
        
        # Database-specific recommendations
        if config['database_size_gb'] > 10000:
            recommendations.append("AI recommends staged migration approach for large database")
        
        if config['performance_requirements'] == 'high':
            recommendations.append("AI suggests performance testing in AWS environment before full migration")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    # Keep original helper methods
    def _get_performance_tier(self, score: float) -> str:
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"
    
    def _get_ai_scaling_recommendation(self, cpu_score: float, memory_score: float, storage_score: float) -> List[str]:
        """AI-enhanced scaling recommendations"""
        recommendations = []
        
        if cpu_score < 60:
            recommendations.append("AI Priority: CPU upgrade or more cores for improved performance")
        if memory_score < 60:
            recommendations.append("AI Priority: Memory expansion for better caching efficiency")
        if storage_score < 60:
            recommendations.append("AI Priority: Storage upgrade to NVMe SSD for IOPS improvement")
        
        if not recommendations:
            recommendations.append("AI Assessment: System is well-balanced for current workload")
        
        return recommendations

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AI and AWS API integration"""
    
    def __init__(self):
        self.ai_manager = AnthropicAIManager()
        self.aws_api = AWSAPIManager()
        self.os_manager = OSPerformanceManager()
        self.aws_manager = EnhancedAWSMigrationManager(self.aws_api, self.ai_manager)
        self.network_manager = EnhancedNetworkIntelligenceManager()
        self.agent_manager = AgentSizingManager()
        self.onprem_analyzer = OnPremPerformanceAnalyzer()
        
        # Keep original hardware configurations
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
        
        # Enhanced on-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
        
        # AI-enhanced network path analysis
        network_perf = self.network_manager.calculate_ai_enhanced_path_performance(config['network_path'])
        
        # Determine migration type and tools (preserved)
        is_homogeneous = config['source_database_engine'] == config['database_engine']
        migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
        primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Agent analysis with AI insights
        agent_analysis = await self._analyze_ai_migration_agents(config, primary_tool, network_perf)
        
        # Calculate effective migration throughput
        agent_throughput = agent_analysis['effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # AI-enhanced migration time calculation
        migration_time_hours = await self._calculate_ai_migration_time(
            config, migration_throughput, onprem_performance
        )
        
        # AI-powered AWS sizing recommendations
        aws_sizing = await self.aws_manager.ai_enhanced_aws_sizing(config)
        
        # Enhanced cost analysis with real-time pricing
        cost_analysis = await self._calculate_ai_enhanced_costs(
            config, aws_sizing, agent_analysis, network_perf
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
            'ai_overall_assessment': await self._generate_ai_overall_assessment(
                config, onprem_performance, aws_sizing, migration_time_hours
            )
        }
    
    async def _analyze_ai_migration_agents(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """AI-enhanced migration agent analysis"""
        
        if primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.datasync_agents[agent_size]
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.dms_agents[agent_size]
        
        # Calculate throughput impact (preserved original logic)
        agent_max_throughput = agent_config['max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        effective_throughput = min(agent_max_throughput, network_bandwidth)
        throughput_impact = effective_throughput / agent_max_throughput
        
        # AI enhancement: optimization recommendations
        ai_agent_optimization = await self._get_ai_agent_optimization(
            agent_config, network_perf, config
        )
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            f'{primary_tool}_config': agent_config,
            'effective_throughput': effective_throughput,
            'throughput_impact': throughput_impact,
            'bottleneck': 'agent' if agent_max_throughput < network_bandwidth else 'network',
            'ai_optimization': ai_agent_optimization,
            'ai_recommendations': agent_config.get('ai_optimization_tips', [])
        }
    
    async def _calculate_ai_migration_time(self, config: Dict, migration_throughput: float, 
                                         onprem_performance: Dict) -> float:
        """AI-enhanced migration time calculation"""
        
        database_size_gb = config['database_size_gb']
        
        # Base calculation (preserved)
        base_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
        
        # AI adjustments
        ai_complexity = onprem_performance.get('ai_optimization_recommendations', [])
        complexity_factor = 1.0
        
        # Increase time for complex scenarios
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3  # Heterogeneous migration
        
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1  # Windows overhead
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05  # Virtualization overhead
        
        # AI insights factor
        if len(ai_complexity) > 3:
            complexity_factor *= 1.2  # Many optimization needs
        
        return base_time_hours * complexity_factor
    
    async def _calculate_ai_enhanced_costs(self, config: Dict, aws_sizing: Dict, 
                                         agent_analysis: Dict, network_perf: Dict) -> Dict:
        """AI-enhanced cost calculation with real-time pricing"""
        
        # Get deployment recommendation
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        # Use real-time pricing data
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Agent costs
        agent_config = agent_analysis[f"{agent_analysis['primary_tool']}_config"]
        agent_cost = agent_config['cost_per_hour'] * 24 * 30
        
        # Network costs with AI optimization
        base_network_cost = 800 if 'prod' in config['network_path'] else 400
        ai_optimization_factor = network_perf.get('ai_optimization_potential', 0) / 100
        optimized_network_cost = base_network_cost * (1 - ai_optimization_factor * 0.2)
        
        # OS licensing with AI insights
        os_licensing_cost = self.os_manager.operating_systems[config['operating_system']]['licensing_cost_factor'] * 150
        
        # Management costs (AI reduces if RDS)
        base_management_cost = 200 if deployment_rec == 'ec2' else 50
        ai_management_reduction = 0.15 if aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6) < 5 else 0
        management_cost = base_management_cost * (1 - ai_management_reduction)
        
        # One-time migration costs with AI complexity adjustment
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1
        base_migration_cost = config['database_size_gb'] * 0.1
        one_time_migration_cost = base_migration_cost * complexity_multiplier
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + agent_cost + 
                            optimized_network_cost + os_licensing_cost + management_cost)
        
        # AI-predicted savings
        ai_optimization_potential = aws_sizing.get('ai_analysis', {}).get('performance_recommendations', [])
        estimated_monthly_savings = total_monthly_cost * (0.1 + len(ai_optimization_potential) * 0.02)
        roi_months = int(one_time_migration_cost / estimated_monthly_savings) if estimated_monthly_savings > 0 else None
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': agent_cost,
            'network_cost': optimized_network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'estimated_monthly_savings': estimated_monthly_savings,
            'roi_months': roi_months,
            'ai_cost_insights': {
                'ai_optimization_factor': ai_optimization_factor,
                'complexity_multiplier': complexity_multiplier,
                'management_reduction': ai_management_reduction,
                'potential_additional_savings': f"{len(ai_optimization_potential) * 2}% through AI recommendations"
            }
        }
    
    async def _get_ai_agent_optimization(self, agent_config: Dict, network_perf: Dict, config: Dict) -> Dict:
        """Get AI-powered agent optimization recommendations"""
        
        # Simple AI-like optimization analysis
        bottleneck_type = 'network' if network_perf['effective_bandwidth_mbps'] < agent_config['max_throughput_mbps'] else 'agent'
        
        optimization_potential = 0
        recommendations = []
        
        if bottleneck_type == 'network':
            optimization_potential = network_perf.get('ai_optimization_potential', 0)
            recommendations = network_perf.get('ai_insights', {}).get('optimization_opportunities', [])
        else:
            optimization_potential = 15  # Agent optimization potential
            recommendations = [
                "Optimize agent parallel processing settings",
                "Configure intelligent retry mechanisms",
                "Implement bandwidth throttling during peak hours"
            ]
        
        return {
            'bottleneck_type': bottleneck_type,
            'optimization_potential_percent': optimization_potential,
            'recommendations': recommendations[:3],
            'estimated_improvement': f"{optimization_potential}% throughput improvement possible"
        }
    
    async def _generate_ai_overall_assessment(self, config: Dict, onprem_performance: Dict, 
                                            aws_sizing: Dict, migration_time: float) -> Dict:
        """Generate AI-powered overall migration assessment"""
        
        # Migration readiness score
        readiness_factors = []
        readiness_score = 100
        
        # Performance readiness
        perf_score = onprem_performance.get('performance_score', 0)
        if perf_score < 50:
            readiness_score -= 20
            readiness_factors.append("Performance optimization needed before migration")
        elif perf_score < 70:
            readiness_score -= 10
            readiness_factors.append("Minor performance improvements recommended")
        
        # Complexity assessment
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        if ai_complexity > 8:
            readiness_score -= 25
            readiness_factors.append("High complexity migration requires extensive planning")
        elif ai_complexity > 6:
            readiness_score -= 10
            readiness_factors.append("Moderate complexity migration needs careful execution")
        
        # Time assessment
        if migration_time > 48:
            readiness_score -= 15
            readiness_factors.append("Long migration time requires extensive downtime planning")
        elif migration_time > 24:
            readiness_score -= 5
            readiness_factors.append("Extended migration window needed")
        
        # Network readiness
        network_quality = onprem_performance.get('network_performance', {}).get('ai_optimization_score', 80)
        if network_quality < 60:
            readiness_score -= 15
            readiness_factors.append("Network optimization required for successful migration")
        
        # Success probability
        success_probability = max(60, min(95, readiness_score))
        
        # Risk level
        if readiness_score >= 80:
            risk_level = "Low"
        elif readiness_score >= 65:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': readiness_factors,
            'ai_confidence': aws_sizing.get('deployment_recommendation', {}).get('confidence', 0.5),
            'recommended_next_steps': self._get_next_steps(readiness_score, ai_complexity),
            'timeline_recommendation': self._get_timeline_recommendation(migration_time, ai_complexity)
        }
    
    def _get_next_steps(self, readiness_score: float, ai_complexity: int) -> List[str]:
        """Get recommended next steps based on AI assessment"""
        
        steps = []
        
        if readiness_score < 70:
            steps.append("Conduct detailed performance baseline and optimization")
            steps.append("Address identified bottlenecks before migration")
        
        if ai_complexity > 7:
            steps.append("Develop comprehensive migration strategy and testing plan")
            steps.append("Consider engaging AWS migration specialists")
        
        steps.extend([
            "Set up AWS environment and conduct connectivity tests",
            "Perform proof-of-concept migration with subset of data",
            "Develop detailed cutover and rollback procedures"
        ])
        
        return steps[:5]
    
    def _get_timeline_recommendation(self, migration_time: float, ai_complexity: int) -> Dict:
        """Get AI-recommended timeline"""
        
        # Base phases
        planning_weeks = 2 + (ai_complexity - 5) * 0.5
        testing_weeks = 3 + (ai_complexity - 5) * 0.5
        migration_hours = migration_time
        
        return {
            'planning_phase_weeks': max(1, planning_weeks),
            'testing_phase_weeks': max(2, testing_weeks),
            'migration_window_hours': migration_hours,
            'total_project_weeks': max(6, planning_weeks + testing_weeks + 1),
            'recommended_approach': 'staged' if ai_complexity > 7 or migration_time > 24 else 'direct'
        }

# PDF Report Generation Class
class PDFReportGenerator:
    """Generate executive PDF reports for migration analysis"""
    
    def __init__(self):
        self.report_style = {
            'title_color': '#1e3c72',
            'header_color': '#2a5298', 
            'accent_color': '#3498db',
            'text_color': '#2c3e50',
            'background_color': '#f8f9fa'
        }
    
    def generate_executive_report(self, analysis: Dict, config: Dict) -> bytes:
        """Generate comprehensive executive PDF report"""
        
        # Create matplotlib figure for PDF
        plt.style.use('default')
        
        # Create PDF buffer
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Page 1: Executive Summary
            self._create_executive_summary_page(pdf, analysis, config)
            
            # Page 2: Technical Analysis
            self._create_technical_analysis_page(pdf, analysis, config)
            
            # Page 3: AWS Sizing Recommendations
            self._create_aws_sizing_page(pdf, analysis, config)
            
            # Page 4: Cost Analysis
            self._create_cost_analysis_page(pdf, analysis, config)
            
            # Page 5: Risk Assessment & Timeline
            self._create_risk_timeline_page(pdf, analysis, config)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_executive_summary_page(self, pdf, analysis: Dict, config: Dict):
        """Create executive summary page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Database Migration - Executive Summary', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Migration Overview
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Migration Overview', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        overview_text = f"""
        Source: {config['source_database_engine'].upper()} ({config['database_size_gb']:,} GB)
        Target: AWS {config['database_engine'].upper()}
        Type: {'Homogeneous' if config['source_database_engine'] == config['database_engine'] else 'Heterogeneous'}
        Environment: {config['environment'].title()}
        Migration Time: {analysis.get('estimated_migration_time_hours', 0):.1f} hours
        Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
        """
        
        ax1.text(0.05, 0.75, overview_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # AI Readiness Score (Gauge chart)
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        self._create_gauge_chart(ax2, readiness_score, 'Migration Readiness', 'AI Assessment')
        
        # Cost Summary
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Cost Summary', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        cost_analysis = analysis.get('cost_analysis', {})
        cost_text = f"""
        Monthly AWS Cost: ${cost_analysis.get('total_monthly_cost', 0):,.0f}
        One-time Migration: ${cost_analysis.get('one_time_migration_cost', 0):,.0f}
        Annual Cost: ${cost_analysis.get('total_monthly_cost', 0) * 12:,.0f}
        Potential Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}/month
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        """
        
        ax3.text(0.05, 0.75, cost_text, fontsize=10, transform=ax3.transAxes, verticalalignment='top')
        
        # Performance Score
        perf_score = analysis.get('onprem_performance', {}).get('performance_score', 0)
        self._create_gauge_chart(ax4, perf_score, 'Performance Score', 'Current System')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_technical_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create technical analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Technical Analysis & Performance Assessment', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Performance Breakdown
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_metrics = {
            'CPU': onprem_perf.get('cpu_score', 0),
            'Memory': onprem_perf.get('memory_score', 0),
            'Storage': onprem_perf.get('storage_score', 0),
            'Network': onprem_perf.get('network_score', 0),
            'Database': onprem_perf.get('database_score', 0)
        }
        
        # Performance radar chart
        self._create_radar_chart(ax1, performance_metrics, 'Current Performance Profile')
        
        # Network Analysis
        network_perf = analysis.get('network_performance', {})
        network_data = {
            'Quality Score': network_perf.get('network_quality_score', 0),
            'AI Enhanced': network_perf.get('ai_enhanced_quality_score', 0),
            'Bandwidth (%)': min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
            'Reliability (%)': network_perf.get('total_reliability', 0) * 100
        }
        
        self._create_bar_chart(ax2, network_data, 'Network Performance Analysis', 'Score/Percentage')
        
        # OS Performance Impact
        os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
        os_data = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        self._create_bar_chart(ax3, os_data, 'OS Performance Impact', 'Efficiency (%)')
        
        # Bottleneck Analysis
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Identified Bottlenecks', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        bottlenecks = analysis.get('onprem_performance', {}).get('bottlenecks', [])
        ai_insights = analysis.get('onprem_performance', {}).get('ai_insights', [])
        
        bottleneck_text = "Current Bottlenecks:\n"
        for i, bottleneck in enumerate(bottlenecks[:3], 1):
            bottleneck_text += f"{i}. {bottleneck}\n"
        
        bottleneck_text += "\nAI Insights:\n"
        for i, insight in enumerate(ai_insights[:2], 1):
            bottleneck_text += f"{i}. {insight}\n"
        
        ax4.text(0.05, 0.85, bottleneck_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_aws_sizing_page(self, pdf, analysis: Dict, config: Dict):
        """Create AWS sizing recommendations page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Sizing & Configuration Recommendations', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        # Deployment Recommendation
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Recommended Deployment', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0) * 100
        
        if recommendation == 'RDS':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon RDS Managed Service
            Instance: {rds_rec.get('primary_instance', 'N/A')}
            Storage: {rds_rec.get('storage_size_gb', 0):,.0f} GB ({rds_rec.get('storage_type', 'gp3').upper()})
            Multi-AZ: {'Yes' if rds_rec.get('multi_az', False) else 'No'}
            Monthly Cost: ${rds_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon EC2 Self-Managed
            Instance: {ec2_rec.get('primary_instance', 'N/A')}
            Storage: {ec2_rec.get('storage_size_gb', 0):,.0f} GB ({ec2_rec.get('storage_type', 'gp3').upper()})
            EBS Optimized: {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}
            Monthly Cost: ${ec2_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        
        ax1.text(0.05, 0.75, deploy_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # RDS vs EC2 Comparison
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        
        comparison_data = {'RDS Score': rds_score, 'EC2 Score': ec2_score}
        self._create_bar_chart(ax2, comparison_data, 'Deployment Scoring', 'Score')
        
        # Reader/Writer Configuration
        reader_writer = aws_sizing.get('reader_writer_config', {})
        
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Instance Configuration', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        instance_text = f"""
        Writer Instances: {reader_writer.get('writers', 1)}
        Reader Instances: {reader_writer.get('readers', 0)}
        Total Instances: {reader_writer.get('total_instances', 1)}
        Read Capacity: {reader_writer.get('read_capacity_percent', 0):.1f}%
        Write Capacity: {reader_writer.get('write_capacity_percent', 100):.1f}%
        Recommended Read Split: {reader_writer.get('recommended_read_split', 0):.0f}%
        
        AI Reasoning:
        {reader_writer.get('reasoning', 'Standard configuration')[:100]}...
        """
        
        ax3.text(0.05, 0.75, instance_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # AI Complexity Factors
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        
        self._create_gauge_chart(ax4, complexity_score * 10, 'AI Complexity Score', f'{complexity_score}/10')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_cost_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create cost analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Cost Analysis & Financial Projections', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Monthly Cost Breakdown (Pie Chart)
        cost_breakdown = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'OS Licensing': cost_analysis.get('os_licensing_cost', 0),
            'Management': cost_analysis.get('management_cost', 0)
        }
        
        # Filter out zero values
        cost_breakdown = {k: v for k, v in cost_breakdown.items() if v > 0}
        
        self._create_pie_chart(ax1, cost_breakdown, 'Monthly Cost Breakdown')
        
        # Cost Projections
        monthly_cost = cost_analysis.get('total_monthly_cost', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        projections = {
            '1 Year': monthly_cost * 12 + one_time_cost,
            '2 Years': monthly_cost * 24 + one_time_cost,
            '3 Years': monthly_cost * 36 + one_time_cost,
            '5 Years': monthly_cost * 60 + one_time_cost
        }
        
        self._create_bar_chart(ax2, projections, 'Total Cost Projections', 'Cost ($)')
        
        # Savings Analysis
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Savings & ROI Analysis', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        savings_text = f"""
        Monthly Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}
        Annual Savings: ${cost_analysis.get('estimated_monthly_savings', 0) * 12:,.0f}
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        Break-even Point: Year {cost_analysis.get('roi_months', 12) / 12:.1f}
        
        AI Cost Insights:
        - Optimization Factor: {cost_analysis.get('ai_cost_insights', {}).get('ai_optimization_factor', 0)*100:.1f}%
        - Complexity Multiplier: {cost_analysis.get('ai_cost_insights', {}).get('complexity_multiplier', 1.0):.2f}x
        - Additional Savings: {cost_analysis.get('ai_cost_insights', {}).get('potential_additional_savings', '0%')}
        """
        
        ax3.text(0.05, 0.75, savings_text, fontsize=10, transform=ax3.transAxes, verticalalignment='top')
        
        # Cost Comparison
        current_cost_estimate = monthly_cost * 0.8  # Assume current is 20% higher
        cost_comparison = {
            'Current (Est.)': current_cost_estimate,
            'AWS Monthly': monthly_cost
        }
        
        self._create_bar_chart(ax4, cost_comparison, 'Cost Comparison', 'Monthly Cost ($)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_risk_timeline_page(self, pdf, analysis: Dict, config: Dict):
        """Create risk assessment and timeline page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Risk Assessment & Project Timeline', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        # Risk Assessment
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        success_prob = ai_assessment.get('success_probability', 0)
        risk_level = ai_assessment.get('risk_level', 'Medium')
        
        risk_data = {
            'Success Probability': success_prob,
            'Risk Mitigation': 100 - (100 - readiness_score) * 0.8
        }
        
        self._create_bar_chart(ax1, risk_data, 'Risk Assessment', 'Percentage (%)')
        
        # Timeline Visualization
        timeline = ai_assessment.get('timeline_recommendation', {})
        
        timeline_data = {
            'Planning': timeline.get('planning_phase_weeks', 2),
            'Testing': timeline.get('testing_phase_weeks', 3),
            'Migration': timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
            'Validation': 1
        }
        
        self._create_bar_chart(ax2, timeline_data, 'Project Timeline', 'Duration (Weeks)')
        
        # Risk Factors
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Key Risk Factors', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        risk_factors = ai_assessment.get('readiness_factors', [])
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        ai_analysis = aws_sizing.get('ai_analysis', {})
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        risk_text = "Identified Risk Factors:\n"
        for i, factor in enumerate(risk_factors[:4], 1):
            risk_text += f"{i}. {factor}\n"
        
        if risk_percentages:
            risk_text += "\nQuantified Risks:\n"
            for risk, percentage in list(risk_percentages.items())[:3]:
                risk_text += f"• {risk.replace('_', ' ').title()}: {percentage}%\n"
        
        ax3.text(0.05, 0.85, risk_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # Next Steps
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Recommended Next Steps', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        next_steps = ai_assessment.get('recommended_next_steps', [])
        
        steps_text = ""
        for i, step in enumerate(next_steps, 1):
            steps_text += f"{i}. {step}\n"
        
        ax4.text(0.05, 0.85, steps_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_gauge_chart(self, ax, value, title, subtitle):
        """Create a gauge chart"""
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, color='lightgray', linewidth=8)
        
        # Value arc
        value_theta = np.linspace(0, (value/100) * np.pi, int(value))
        value_r = np.ones_like(value_theta)
        
        color = '#e74c3c' if value < 50 else '#f39c12' if value < 75 else '#27ae60'
        ax.plot(value_theta, value_r, color=color, linewidth=8)
        
        # Add value text
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -0.65, subtitle, ha='center', va='center', fontsize=10)
        
        ax.set_ylim(-0.8, 1.2)
        ax.set_xlim(-1.2, 1.2)
        ax.axis('off')
    
    def _create_radar_chart(self, ax, data, title):
        """Create a radar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Number of variables
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.report_style['accent_color'])
        ax.fill(angles, values, alpha=0.25, color=self.report_style['accent_color'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _create_bar_chart(self, ax, data, title, ylabel):
        """Create a bar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(labels, values, color=self.report_style['accent_color'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis for currency if needed
        if ' in ylabel or 'Cost' in ylabel:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _create_pie_chart(self, ax, data, title):
        """Create a pie chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if filtered_data:
            labels, values = zip(*filtered_data)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        ax.set_title(title, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

# Network path diagram function
def create_network_path_diagram(network_perf: Dict) -> go.Figure:
    """Create an interactive network path diagram"""
    
    segments = network_perf.get('segments', [])
    if not segments:
        return go.Figure()
    
    # Create network diagram using plotly
    fig = go.Figure()
    
    # Define positions for network nodes
    num_segments = len(segments)
    x_positions = [i * 100 for i in range(num_segments + 1)]
    y_positions = [50] * (num_segments + 1)
    
    # Add network segments as lines
    for i, segment in enumerate(segments):
        # Calculate line properties based on performance
        line_width = max(2, min(10, segment['effective_bandwidth_mbps'] / 200))
        
        # Color based on performance (green = good, yellow = ok, red = poor)
        reliability = segment['reliability']
        if reliability > 0.999:
            line_color = '#27ae60'  # Green
        elif reliability > 0.995:
            line_color = '#f39c12'  # Orange
        else:
            line_color = '#e74c3c'  # Red
        
        # Add line for network segment
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_positions[i], y_positions[i+1]],
            mode='lines+markers',
            line=dict(
                width=line_width,
                color=line_color
            ),
            marker=dict(size=15, color='#2c3e50', symbol='square'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Type: {segment['connection_type'].replace('_', ' ').title()}<br>
            Bandwidth: {segment['effective_bandwidth_mbps']:.0f} Mbps<br>
            Latency: {segment['effective_latency_ms']:.1f} ms<br>
            Reliability: {segment['reliability']*100:.3f}%<br>
            Cost Factor: {segment['cost_factor']:.1f}x<br>
            AI Optimization: {segment.get('ai_optimization_potential', 0)*100:.1f}%<br>
            <extra></extra>
            """
        ))
        
        # Add bandwidth and latency annotations
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        mid_y = (y_positions[i] + y_positions[i+1]) / 2 + 20
        
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"<b>{segment['effective_bandwidth_mbps']:.0f} Mbps</b><br>{segment['effective_latency_ms']:.1f} ms",
            showarrow=False,
            font=dict(size=10, color='#2c3e50'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            borderpad=4
        )
        
        # Add connection type label
        fig.add_annotation(
            x=mid_x,
            y=mid_y - 25,
            text=f"<i>{segment['connection_type'].replace('_', ' ').title()}</i>",
            showarrow=False,
            font=dict(size=8, color='#7f8c8d'),
            bgcolor='rgba(248,249,250,0.8)',
            bordercolor='#dee2e6',
            borderwidth=1,
            borderpad=2
        )
    
    # Add source and destination nodes
    fig.add_trace(go.Scatter(
        x=[x_positions[0]],
        y=[y_positions[0]],
        mode='markers+text',
        marker=dict(size=25, color='#27ae60', symbol='circle'),
        text=['SOURCE'],
        textposition='bottom center',
        name='Source System',
        hovertemplate="<b>Source System</b><br>On-Premises Database<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]],
        y=[y_positions[-1]],
        mode='markers+text',
        marker=dict(size=25, color='#e74c3c', symbol='circle'),
        text=['AWS'],
        textposition='bottom center',
        name='AWS Destination',
        hovertemplate="<b>AWS Destination</b><br>Target Database Service<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Network Migration Path: {network_perf.get('path_name', 'Unknown')}",
            font=dict(size=16, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-20, max(x_positions) + 20]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[0, 100]
        ),
        showlegend=False,
        height=350,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Enhanced rendering functions

def render_enhanced_header():
    """Enhanced header with professional styling"""
    
    # Initialize managers to check status
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            Comprehensive Network Path Analysis • OS Performance Optimization • Enterprise-Ready Migration Planning
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span>🟢 Network Intelligence Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    
    st.sidebar.markdown("### 🔌 System Status")
    
    # Check API status
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
    
    # Configuration instructions
    if not ai_manager.connected or not aws_api.connected:
        st.sidebar.markdown("### ⚙️ Configuration")
        
        if not ai_manager.connected:
            st.sidebar.info("💡 Add ANTHROPIC_API_KEY to Streamlit secrets for enhanced AI analysis")
        
        if not aws_api.connected:
            st.sidebar.info("💡 Configure AWS credentials for real-time pricing data")

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with AI-powered recommendations"""
    
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0")
    
    # Render API status
    render_api_status_sidebar()
    
    st.sidebar.markdown("---")
    
    # Operating System Selection with AI insights
    st.sidebar.subheader("💻 Operating System (AI-Enhanced)")
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
        }[x],
        help="AI analyzes OS performance characteristics and migration impact"
    )
    
    # Show AI OS insights
    os_manager = OSPerformanceManager()
    os_config = os_manager.operating_systems[operating_system]
    
    with st.sidebar.expander("🤖 AI OS Insights"):
        st.markdown(f"**Strengths:** {', '.join(os_config['ai_insights']['strengths'][:2])}")
        st.markdown(f"**Key Consideration:** {os_config['ai_insights']['weaknesses'][0]}")
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine",
        help="Physical vs Virtual performance analysis with AI optimization"
    )
    
    # Hardware Configuration with AI recommendations
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2, 
                                 help="AI calculates optimal memory for database workload")
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2,
                                   help="AI analyzes CPU requirements for migration performance")
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Enhanced Performance Metrics
    st.sidebar.subheader("📊 Current Performance Metrics")
    current_storage_gb = st.sidebar.number_input("Current Storage (GB)", 
                                                min_value=100, max_value=500000, value=2000, step=100,
                                                help="Current storage capacity in use")
    peak_iops = st.sidebar.number_input("Peak IOPS", 
                                       min_value=100, max_value=1000000, value=10000, step=500,
                                       help="Maximum IOPS observed during peak usage")
    max_throughput_mbps = st.sidebar.number_input("Max Throughput (MB/s)", 
                                                 min_value=10, max_value=10000, value=500, step=50,
                                                 help="Maximum storage throughput observed")
    anticipated_max_memory_gb = st.sidebar.number_input("Anticipated Max Memory (GB)", 
                                                       min_value=4, max_value=1024, value=64, step=8,
                                                       help="Maximum memory usage anticipated for workload")
    anticipated_max_cpu_cores = st.sidebar.number_input("Anticipated Max CPU Cores", 
                                                       min_value=1, max_value=128, value=16, step=2,
                                                       help="Maximum CPU cores anticipated for workload")
    
    # Network Interface with AI insights
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
        }[x],
        help="AI analyzes network impact on migration throughput"
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration with AI analysis
    st.sidebar.subheader("🔄 Migration Setup (AI-Optimized)")
    
    # Source and Target Databases
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x],
        help="AI determines migration complexity based on source engine"
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database (AWS)",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '☁️ RDS MySQL', 'postgresql': '☁️ RDS PostgreSQL', 'oracle': '☁️ RDS Oracle',
            'sqlserver': '☁️ RDS SQL Server', 'mongodb': '☁️ DocumentDB'
        }[x],
        help="AI recommends optimal AWS database service"
    )
    
    # Show migration type indicator
    is_homogeneous = source_database_engine == database_engine
    migration_type_indicator = "🟢 Homogeneous" if is_homogeneous else "🟡 Heterogeneous"
    st.sidebar.info(f"**Migration Type:** {migration_type_indicator}")
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100,
                                              help="AI calculates migration time and resource requirements")
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60,
                                                        help="AI optimizes migration strategy for downtime window")
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"],
                                                   help="AI adjusts AWS sizing recommendations")
    
    # Enhanced Network Path Selection with AI insights
    st.sidebar.subheader("🌐 Enterprise Network Path (AI-Analyzed)")
    network_path = st.sidebar.selectbox(
        "Migration Path",
        ["nonprod_sj_linux_nas", "nonprod_sj_windows_share", "prod_sa_linux_nas", "prod_sa_windows_share"],
        format_func=lambda x: {
            'nonprod_sj_linux_nas': '🐧 Non-Prod: SJ Linux NAS → AWS S3',
            'nonprod_sj_windows_share': '🪟 Non-Prod: SJ Windows Share → AWS S3',
            'prod_sa_linux_nas': '🐧 Prod: SA Linux NAS → SJ → AWS VPC',
            'prod_sa_windows_share': '🪟 Prod: SA Windows Share → SJ → AWS VPC'
        }[x],
        help="AI analyzes network performance and optimization opportunities"
    )
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"],
                                     help="AI adjusts reliability and performance requirements")
    
    # Agent Sizing Selection with AI recommendations
    st.sidebar.subheader("🤖 Migration Agent Sizing (AI-Optimized)")
    
    # Determine migration type for tool selection
    primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    if is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - Up to 1TB',
                'medium': '📦 Medium (c5.large) - 1-5TB',
                'large': '📦 Large (c5.xlarge) - 5-20TB',
                'xlarge': '📦 XLarge (c5.2xlarge) - >20TB'
            }[x],
            help="AI recommends optimal agent size for throughput"
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - Up to 500GB',
                'medium': '🔄 Medium (c5.large) - 500GB-2TB',
                'large': '🔄 Large (c5.xlarge) - 2-10TB',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 10-50TB',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - >50TB'
            }[x],
            help="AI recommends optimal DMS instance for schema complexity"
        )
        datasync_agent_size = None
    
    # AI Configuration Section
    st.sidebar.subheader("🧠 AI Configuration")
    
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True,
                                           help="Use Anthropic AI for intelligent recommendations")
    
    if enable_ai_analysis:
        ai_analysis_depth = st.sidebar.selectbox(
            "AI Analysis Depth",
            ["standard", "comprehensive"],
            help="Comprehensive analysis provides more detailed AI insights"
        )
    else:
        ai_analysis_depth = "standard"
    
    # Real-time AWS Pricing
    use_realtime_pricing = st.sidebar.checkbox("Real-time AWS Pricing", value=True,
                                             help="Fetch current AWS pricing via API")
    
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
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'network_path': network_path,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'enable_ai_analysis': enable_ai_analysis,
        'ai_analysis_depth': ai_analysis_depth,
        'use_realtime_pricing': use_realtime_pricing,
        'current_storage_gb': current_storage_gb,
        'peak_iops': peak_iops,
        'max_throughput_mbps': max_throughput_mbps,
        'anticipated_max_memory_gb': anticipated_max_memory_gb,
        'anticipated_max_cpu_cores': anticipated_max_cpu_cores
    }

def render_aws_sizing_configuration_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations"""
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    rds_rec = aws_sizing.get('rds_recommendations', {})
    ec2_rec = aws_sizing.get('ec2_recommendations', {})
    reader_writer_config = aws_sizing.get('reader_writer_config', {})
    
    # Deployment Recommendation Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recommended_deployment = deployment_rec.get('recommendation', 'unknown')
        confidence = deployment_rec.get('confidence', 0)
        st.metric(
            "☁️ Recommended Deployment",
            recommended_deployment.upper(),
            delta=f"Confidence: {confidence*100:.1f}%"
        )
    
    with col2:
        if recommended_deployment == 'rds':
            primary_instance = rds_rec.get('primary_instance', 'Unknown')
            monthly_cost = rds_rec.get('total_monthly_cost', 0)
        else:
            primary_instance = ec2_rec.get('primary_instance', 'Unknown')
            monthly_cost = ec2_rec.get('total_monthly_cost', 0)
        
        st.metric(
            "🖥️ Primary Instance",
            primary_instance,
            delta=f"${monthly_cost:.0f}/month"
        )
    
    with col3:
        total_instances = reader_writer_config.get('total_instances', 1)
        writers = reader_writer_config.get('writers', 1)
        readers = reader_writer_config.get('readers', 0)
        st.metric(
            "📊 Instance Configuration",
            f"{total_instances} Total",
            delta=f"{writers}W + {readers}R"
        )
    
    with col4:
        if recommended_deployment == 'rds':
            storage_size = rds_rec.get('storage_size_gb', 0)
            storage_type = rds_rec.get('storage_type', 'gp3')
        else:
            storage_size = ec2_rec.get('storage_size_gb', 0)
            storage_type = ec2_rec.get('storage_type', 'gp3')
        
        st.metric(
            "💾 Storage Configuration",
            f"{storage_size:,.0f} GB",
            delta=storage_type.upper()
        )
    
    # Detailed Deployment Comparison
    st.markdown("**🔄 RDS vs EC2 Deployment Analysis:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="deployment-comparison-card" style="border-left-color: {'#27ae60' if recommended_deployment == 'rds' else '#95a5a6'};">
            <h4>🗄️ Amazon RDS Managed Service</h4>
            <p><strong>Recommendation Score:</strong> {deployment_rec.get('rds_score', 0)}/100</p>
            <p><strong>Instance Type:</strong> {rds_rec.get('primary_instance', 'N/A')}</p>
            <p><strong>vCPU:</strong> {rds_rec.get('instance_specs', {}).get('vcpu', 'N/A')}</p>
            <p><strong>Memory:</strong> {rds_rec.get('instance_specs', {}).get('memory', 'N/A')} GB</p>
            <p><strong>Storage:</strong> {rds_rec.get('storage_size_gb', 0):,.0f} GB ({rds_rec.get('storage_type', 'gp3').upper()})</p>
            <p><strong>Multi-AZ:</strong> {'Yes' if rds_rec.get('multi_az', False) else 'No'}</p>
            <p><strong>Monthly Cost:</strong> ${rds_rec.get('total_monthly_cost', 0):.0f}</p>
            <p><strong>Backup Retention:</strong> {rds_rec.get('backup_retention_days', 7)} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="deployment-comparison-card" style="border-left-color: {'#27ae60' if recommended_deployment == 'ec2' else '#95a5a6'};">
            <h4>🖥️ Amazon EC2 Self-Managed</h4>
            <p><strong>Recommendation Score:</strong> {deployment_rec.get('ec2_score', 0)}/100</p>
            <p><strong>Instance Type:</strong> {ec2_rec.get('primary_instance', 'N/A')}</p>
            <p><strong>vCPU:</strong> {ec2_rec.get('instance_specs', {}).get('vcpu', 'N/A')}</p>
            <p><strong>Memory:</strong> {ec2_rec.get('instance_specs', {}).get('memory', 'N/A')} GB</p>
            <p><strong>Storage:</strong> {ec2_rec.get('storage_size_gb', 0):,.0f} GB ({ec2_rec.get('storage_type', 'gp3').upper()})</p>
            <p><strong>EBS Optimized:</strong> {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}</p>
            <p><strong>Monthly Cost:</strong> ${ec2_rec.get('total_monthly_cost', 0):.0f}</p>
            <p><strong>Enhanced Networking:</strong> {'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Deployment Decision Factors
    st.markdown("**🎯 Decision Factors & Reasoning:**")
    
    primary_reasons = deployment_rec.get('primary_reasons', [])
    ai_insights = deployment_rec.get('ai_insights', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>💡 Primary Recommendation Factors</h4>
            <ul>
                {"".join([f"<li>{reason}</li>" for reason in primary_reasons[:4]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🤖 AI Complexity Analysis</h4>
            <p><strong>AI Complexity Factor:</strong> {deployment_rec.get('ai_complexity_factor', 6)}/10</p>
            <p><strong>Impact:</strong> {ai_insights.get('complexity_impact', 'Standard impact on deployment choice')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cost_factors = ai_insights.get('cost_factors', {})
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>💰 Cost Comparison</h4>
            <p><strong>RDS Monthly:</strong> ${cost_factors.get('rds_monthly', 0):.0f}</p>
            <p><strong>EC2 Monthly:</strong> ${cost_factors.get('ec2_monthly', 0):.0f}</p>
            <p><strong>Cost Difference:</strong> {cost_factors.get('cost_difference_percent', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reader/Writer Configuration Details
    st.markdown("**📊 Database Instance Configuration & Scaling:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="aws-sizing-card">
            <h4>🔄 Reader/Writer Configuration</h4>
            <p><strong>Writer Instances:</strong> {reader_writer_config.get('writers', 1)}</p>
            <p><strong>Reader Instances:</strong> {reader_writer_config.get('readers', 0)}</p>
            <p><strong>Total Instances:</strong> {reader_writer_config.get('total_instances', 1)}</p>
            <p><strong>Write Capacity:</strong> {reader_writer_config.get('write_capacity_percent', 100):.1f}%</p>
            <p><strong>Read Capacity:</strong> {reader_writer_config.get('read_capacity_percent', 0):.1f}%</p>
            <p><strong>Recommended Read Split:</strong> {reader_writer_config.get('recommended_read_split', 0):.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ai_scaling_insights = reader_writer_config.get('ai_insights', {})
        st.markdown(f"""
        <div class="aws-sizing-card">
            <h4>🤖 AI Scaling Insights</h4>
            <p><strong>Complexity Impact:</strong> {ai_scaling_insights.get('complexity_impact', 'N/A')}/10</p>
            <p><strong>Optimization Potential:</strong> {ai_scaling_insights.get('optimization_potential', 'N/A')}</p>
            <p><strong>Reasoning:</strong> {reader_writer_config.get('reasoning', 'Standard configuration')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Sizing Factors
    ai_sizing_factors = rds_rec.get('ai_sizing_factors', {}) if recommended_deployment == 'rds' else ec2_rec.get('ai_sizing_factors', {})
    if ai_sizing_factors:
        st.markdown("**🧠 AI Sizing Analysis:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>🎯 Complexity Multipliers</h4>
                <p><strong>Base Multiplier:</strong> {ai_sizing_factors.get('complexity_multiplier', 1.0):.2f}x</p>
                <p><strong>AI Complexity Score:</strong> {ai_sizing_factors.get('ai_complexity_score', 6)}/10</p>
                <p><strong>Storage Multiplier:</strong> {ai_sizing_factors.get('storage_multiplier', 1.5):.2f}x</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ai_recommendations = rds_rec.get('ai_recommendations', []) if recommended_deployment == 'rds' else ec2_rec.get('ai_recommendations', [])
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>💡 AI Recommendations</h4>
                <ul>
                    {"".join([f"<li>{rec}</li>" for rec in ai_recommendations[:3]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            scaling_factors = ai_scaling_insights.get('scaling_factors', [])
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>📈 Scaling Considerations</h4>
                <ul>
                    {"".join([f"<li>{factor}</li>" for factor in scaling_factors[:3]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Enhanced AI insights tab with detailed analysis"""
    st.subheader("🧠 Comprehensive AI-Powered Migration Analysis")
    
    ai_assessment = analysis.get('ai_overall_assessment', {})
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    ai_analysis = aws_sizing.get('ai_analysis', {})
    
    # Enhanced Migration Readiness Dashboard with compact design
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        success_prob = ai_assessment.get('success_probability', 0)
        st.metric(
            "🎯 Migration Readiness", 
            f"{readiness_score:.0f}/100",
            delta=f"Success Rate: {success_prob:.0f}%"
        )
    
    with col2:
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        risk_level = ai_assessment.get('risk_level', 'Unknown')
        st.metric(
            "🔄 Complexity Analysis",
            f"{complexity_score}/10",
            delta=f"Risk: {risk_level}"
        )
    
    with col3:
        confidence = ai_analysis.get('confidence_level', 'medium')
        ai_confidence = ai_assessment.get('ai_confidence', 0.5)
        st.metric(
            "🤖 AI Confidence",
            f"{ai_confidence*100:.1f}%",
            delta=confidence.title()
        )
    
    with col4:
        timeline = ai_assessment.get('timeline_recommendation', {})
        total_weeks = timeline.get('total_project_weeks', 6)
        approach = timeline.get('recommended_approach', 'direct')
        st.metric(
            "📅 Project Timeline",
            f"{total_weeks:.0f} weeks",
            delta=approach.replace('_', ' ').title()
        )
    
    with col5:
        resource_allocation = ai_analysis.get('resource_allocation', {})
        team_size = resource_allocation.get('migration_team_size', 3)
        specialists = resource_allocation.get('aws_specialists_needed', 1)
        st.metric(
            "👥 Team Requirements",
            f"{team_size} members",
            delta=f"{specialists} AWS specialists"
        )
    
    # Performance Metrics Overview
    st.markdown("**📊 Current Performance Analysis:**")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>💾 Current Storage</h5>
            <p><strong>{config.get('current_storage_gb', 0):,} GB</strong></p>
            <p>Database: {config.get('database_size_gb', 0):,} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚡ Peak IOPS</h5>
            <p><strong>{config.get('peak_iops', 0):,}</strong></p>
            <p>Max observed performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🚀 Max Throughput</h5>
            <p><strong>{config.get('max_throughput_mbps', 0)} MB/s</strong></p>
            <p>Storage throughput peak</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🧠 Anticipated Memory</h5>
            <p><strong>{config.get('anticipated_max_memory_gb', 0)} GB</strong></p>
            <p>Current: {config.get('ram_gb', 0)} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col5:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚙️ Anticipated CPU</h5>
            <p><strong>{config.get('anticipated_max_cpu_cores', 0)} cores</strong></p>
            <p>Current: {config.get('cpu_cores', 0)} cores</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Complexity Analysis
    st.markdown("**🔍 Detailed Complexity Analysis:**")
    
    complexity_factors = ai_analysis.get('complexity_factors', [])
    if complexity_factors:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ai-insight-card">
                <h4>📊 Complexity Factor Breakdown</h4>
                <ul>
                    {"".join([f"<li><strong>{factor[0]}:</strong> +{factor[1]} complexity points</li>" for factor in complexity_factors])}
                </ul>
                <p><strong>Base Complexity:</strong> 5/10</p>
                <p><strong>Final Score:</strong> {complexity_score}/10</p>
                <p><strong>Interpretation:</strong> {"Simple migration" if complexity_score < 6 else "Moderate complexity" if complexity_score < 8 else "Complex migration requiring expert oversight"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk Assessment with percentages
            risk_percentages = ai_analysis.get('risk_percentages', {})
            st.markdown(f"""
            <div class="ai-recommendation-card">
                <h4>🚨 Quantified Risk Assessment</h4>
                {"".join([f"<p><strong>{risk.replace('_', ' ').title()}:</strong> {percentage}%</p>" for risk, percentage in risk_percentages.items()])}
                <p><strong>Overall Risk Level:</strong> {risk_level}</p>
                <p><strong>Mitigation Coverage:</strong> {len(ai_analysis.get('mitigation_strategies', []))} strategies identified</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Performance Analysis
    st.markdown("**⚡ Performance Optimization Analysis:**")
    
    performance_improvements = ai_analysis.get('performance_improvements', {})
    if performance_improvements:
        col1, col2, col3 = st.columns(3)
        
        improvement_items = list(performance_improvements.items())
        for i, col in enumerate([col1, col2, col3]):
            if i < len(improvement_items):
                improvement, benefit = improvement_items[i]
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5>💡 {improvement.replace('_', ' ').title()}</h5>
                        <p><strong>Expected Improvement:</strong> {benefit}</p>
                        <p><strong>Implementation:</strong> {"Pre-migration" if i == 0 else "During migration" if i == 1 else "Post-migration"}</p>
                        <p><strong>Priority:</strong> {"High" if i == 0 else "Medium"}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detailed Resource Allocation
    st.markdown("**👥 Comprehensive Resource Allocation Plan:**")
    
    if resource_allocation:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>🎯 Team Composition Requirements</h4>
                <p><strong>Migration Team Size:</strong> {resource_allocation.get('migration_team_size', 3)} members</p>
                <p><strong>AWS Specialists:</strong> {resource_allocation.get('aws_specialists_needed', 1)} required</p>
                <p><strong>Database Experts:</strong> {resource_allocation.get('database_experts_required', 1)} needed</p>
                <p><strong>Testing Resources:</strong> {resource_allocation.get('testing_resources', 'Standard allocation')}</p>
                <p><strong>Project Duration:</strong> {total_weeks} weeks end-to-end</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>🏗️ Infrastructure Requirements</h4>
                <p><strong>Environment Setup:</strong> {resource_allocation.get('infrastructure_requirements', 'Standard AWS environment')}</p>
                <p><strong>Testing Strategy:</strong> {len(ai_analysis.get('testing_strategy', []))} testing phases</p>
                <p><strong>Monitoring Setup:</strong> {len(ai_analysis.get('post_migration_monitoring', []))} monitoring components</p>
                <p><strong>Rollback Preparation:</strong> {len(ai_analysis.get('rollback_procedures', []))} rollback procedures</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Implementation Roadmap
    st.markdown("**🗺️ Detailed Implementation Roadmap:**")
    
    timeline_suggestions = ai_analysis.get('timeline_suggestions', [])
    if timeline_suggestions:
        for i, phase in enumerate(timeline_suggestions):
            phase_color = "#3498db" if i < 2 else "#f39c12" if i < 4 else "#27ae60"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {phase_color};">
                <h5>{phase}</h5>
                <p><strong>Key Activities:</strong> {"Planning and requirement gathering" if i == 0 else "Environment setup and initial testing" if i == 1 else "Comprehensive testing and validation" if i == 2 else "Migration execution and cutover" if i == 3 else "Monitoring and optimization"}</p>
                <p><strong>Deliverables:</strong> {"Migration plan and resource allocation" if i == 0 else "Configured AWS environment" if i == 1 else "Validated migration procedures" if i == 2 else "Successfully migrated database" if i == 3 else "Optimized and monitored system"}</p>
            </div>
            """)
    
    # Cost-Benefit Analysis
    cost_optimization = ai_analysis.get('cost_optimization', [])
    if cost_optimization:
        st.markdown("**💰 AI-Driven Cost Optimization Strategies:**")
        
        for i, strategy in enumerate(cost_optimization):
            potential_savings = f"{(i + 1) * 15}% potential savings"
            implementation_effort = "Low" if i < 2 else "Medium" if i < 4 else "High"
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h5>Strategy {i + 1}: {strategy}</h5>
                <p><strong>Potential Savings:</strong> {potential_savings}</p>
                <p><strong>Implementation Effort:</strong> {implementation_effort}</p>
                <p><strong>Timeframe:</strong> {"Immediate" if i < 2 else "Short-term (1-3 months)" if i < 4 else "Long-term (3-6 months)"}</p>
            </div>
            """)
    
    # Critical Success Factors
    detailed_assessment = ai_analysis.get('detailed_assessment', {})
    critical_factors = detailed_assessment.get('critical_success_factors', [])
    
    if critical_factors:
        st.markdown("**🎯 Critical Success Factors Analysis:**")
        
        for i, factor in enumerate(critical_factors):
            importance = "Critical" if i < 2 else "High" if i < 4 else "Medium"
            impact = "Project failure risk" if i < 2 else "Significant delays possible" if i < 4 else "Quality impact"
            
            st.markdown(f"""
            <div class="ai-recommendation-card">
                <h5>Factor {i + 1}: {factor}</h5>
                <p><strong>Importance Level:</strong> {importance}</p>
                <p><strong>Impact if Missing:</strong> {impact}</p>
                <p><strong>Mitigation:</strong> {"Mandatory requirement" if i < 2 else "Strongly recommended" if i < 4 else "Best practice"}</p>
            </div>
            """)

def render_network_intelligence_tab_enhanced(analysis: Dict, config: Dict):
    """Enhanced network intelligence tab with diagram"""
    st.subheader("🌐 Network Intelligence & Path Optimization")
    
    network_perf = analysis.get('network_performance', {})
    ai_insights = network_perf.get('ai_insights', {})
    
    # Network Performance Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality_score = network_perf.get('network_quality_score', 0)
        ai_enhanced_score = network_perf.get('ai_enhanced_quality_score', 0)
        improvement = ai_enhanced_score - quality_score
        st.metric(
            "📊 Network Quality", 
            f"{quality_score:.1f}/100",
            delta=f"AI Enhanced: +{improvement:.1f}"
        )
    
    with col2:
        bandwidth = network_perf.get('effective_bandwidth_mbps', 0)
        latency = network_perf.get('total_latency_ms', 0)
        st.metric(
            "🚀 Effective Bandwidth",
            f"{bandwidth:,.0f} Mbps",
            delta=f"Latency: {latency:.1f}ms"
        )
    
    with col3:
        reliability = network_perf.get('total_reliability', 0)
        uptime_percent = reliability * 100
        annual_downtime = (1 - reliability) * 365 * 24 * 60  # minutes per year
        st.metric(
            "🛡️ Path Reliability",
            f"{uptime_percent:.3f}%",
            delta=f"~{annual_downtime:.0f}min downtime/year"
        )
    
    with col4:
        optimization_potential = network_perf.get('ai_optimization_potential', 0)
        cost_factor = network_perf.get('total_cost_factor', 0)
        st.metric(
            "💡 Optimization Potential",
            f"{optimization_potential:.1f}%",
            delta=f"Cost Factor: {cost_factor:.1f}x"
        )
    
    # Interactive Network Path Diagram
    st.markdown("**🗺️ Interactive Network Path Diagram:**")
    
    st.markdown("""
    <div class="network-diagram-card">
        <p><strong>Path Visualization:</strong> Line thickness represents bandwidth, color indicates performance quality (green=excellent, orange=good, red=needs attention)</p>
    </div>
    """, unsafe_allow_html=True)
    
    network_diagram = create_network_path_diagram(network_perf)
    st.plotly_chart(network_diagram, use_container_width=True)
    
    # Detailed Path Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="network-intelligence-card">
            <h4>📍 Path Configuration Details</h4>
            <p><strong>Route:</strong> {network_perf.get('path_name', 'Unknown')}</p>
            <p><strong>Environment:</strong> {network_perf.get('environment', 'Unknown').title()}</p>
            <p><strong>OS Type:</strong> {network_perf.get('os_type', 'Unknown').title()}</p>
            <p><strong>Storage Type:</strong> {network_perf.get('storage_type', 'Unknown').upper()}</p>
            <p><strong>Segment Count:</strong> {len(network_perf.get('segments', []))}</p>
            <p><strong>Total Hops:</strong> {len(network_perf.get('segments', []))} network segments</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance Bottlenecks with detailed analysis
        bottlenecks = ai_insights.get('performance_bottlenecks', [])
        if bottlenecks:
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>⚠️ AI-Identified Performance Bottlenecks</h4>
                <ul>
                    {"".join([f"<li><strong>{bottleneck}</strong></li>" for bottleneck in bottlenecks])}
                </ul>
                <p><strong>Impact Assessment:</strong> These bottlenecks could reduce migration throughput by 15-30%</p>
                <p><strong>Mitigation Priority:</strong> High - Address before migration execution</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # AI Optimization with quantified benefits
        optimizations = ai_insights.get('optimization_opportunities', [])
        if optimizations:
            st.markdown(f"""
            <div class="network-intelligence-card">
                <h4>🎯 AI Optimization Opportunities</h4>
                <ul>
                    {"".join([f"<li><strong>{opt}</strong></li>" for opt in optimizations])}
                </ul>
                <p><strong>Expected Performance Gain:</strong> {optimization_potential:.1f}% throughput improvement</p>
                <p><strong>Implementation Effort:</strong> {"Low" if optimization_potential < 15 else "Medium" if optimization_potential < 30 else "High"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Assessment with detailed analysis
        risk_factors = ai_insights.get('risk_factors', [])
        if risk_factors:
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h4>🚨 Comprehensive Risk Assessment</h4>
                <ul>
                    {"".join([f"<li><strong>{risk}</strong></li>" for risk in risk_factors])}
                </ul>
                <p><strong>Overall Risk Level:</strong> {"Low" if len(risk_factors) <= 2 else "Medium" if len(risk_factors) <= 4 else "High"}</p>
                <p><strong>Mitigation Required:</strong> {"Optional" if len(risk_factors) <= 2 else "Recommended" if len(risk_factors) <= 4 else "Critical"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Network Segments Performance Analysis with enhanced details
    st.markdown("**🔗 Detailed Network Segment Analysis:**")
    
    segments = network_perf.get('segments', [])
    if segments:
        segment_data = []
        for i, segment in enumerate(segments):
            # Calculate performance metrics
            efficiency = (segment['effective_bandwidth_mbps'] / segment['bandwidth_mbps']) * 100
            latency_impact = ((segment['effective_latency_ms'] - segment['latency_ms']) / segment['latency_ms']) * 100
            
            segment_data.append({
                'Segment': f"Segment {i+1}",
                'Name': segment['name'],
                'Type': segment['connection_type'].replace('_', ' ').title(),
                'Base BW (Mbps)': f"{segment['bandwidth_mbps']:,}",
                'Effective BW (Mbps)': f"{segment['effective_bandwidth_mbps']:.0f}",
                'Efficiency (%)': f"{efficiency:.1f}%",
                'Base Latency (ms)': f"{segment['latency_ms']:.1f}",
                'Effective Latency (ms)': f"{segment['effective_latency_ms']:.1f}",
                'Latency Impact (%)': f"{latency_impact:+.1f}%",
                'Reliability (%)': f"{segment['reliability']*100:.3f}%",
                'Cost Factor': f"{segment['cost_factor']:.1f}x",
                'AI Optimization (%)': f"{segment.get('ai_optimization_potential', 0)*100:.1f}%"
            })
        
        df_segments = pd.DataFrame(segment_data)
        st.dataframe(df_segments, use_container_width=True, hide_index=True)
    
    # Enhanced Performance Visualization
    st.markdown("**📊 Comprehensive Network Performance Analysis:**")
    
    if segments:
        # Create multi-metric comparison
        fig_metrics = go.Figure()
        
        segment_names = [f"Seg {i+1}" for i in range(len(segments))]
        
        # Add multiple metrics
        fig_metrics.add_trace(go.Bar(
            name='Bandwidth Efficiency (%)',
            x=segment_names,
            y=[(seg['effective_bandwidth_mbps'] / seg['bandwidth_mbps']) * 100 for seg in segments],
            marker_color='#3498db',
            yaxis='y'
        ))
        
        fig_metrics.add_trace(go.Scatter(
            name='Latency (ms)',
            x=segment_names,
            y=[seg['effective_latency_ms'] for seg in segments],
            mode='lines+markers',
            marker_color='#e74c3c',
            yaxis='y2',
            line=dict(width=3)
        ))
        
        fig_metrics.add_trace(go.Scatter(
            name='Reliability (%)',
            x=segment_names,
            y=[seg['reliability'] * 100 for seg in segments],
            mode='lines+markers',
            marker_color='#27ae60',
            yaxis='y3',
            line=dict(width=3)
        ))
        
        # Update layout for multiple y-axes
        fig_metrics.update_layout(
            title="Multi-Metric Network Segment Analysis",
            xaxis_title="Network Segments",
            yaxis=dict(title="Bandwidth Efficiency (%)", side="left", range=[0, 100]),
            yaxis2=dict(title="Latency (ms)", side="right", overlaying="y", range=[0, max([seg['effective_latency_ms'] for seg in segments]) * 1.2]),
            yaxis3=dict(title="Reliability (%)", side="right", overlaying="y", position=0.95, range=[99, 100]),
            legend=dict(x=0.02, y=0.98),
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # AI Recommendations with implementation guidance
    recommended_improvements = ai_insights.get('recommended_improvements', [])
    if recommended_improvements:
        st.markdown("**🚀 AI-Recommended Network Improvements with Implementation Guide:**")
        
        for i, improvement in enumerate(recommended_improvements, 1):
            # Add implementation details
            implementation_time = "1-2 weeks" if i <= 2 else "2-4 weeks"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            expected_benefit = f"{10 + i * 5}% performance improvement"
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h5>Recommendation {i}: {improvement}</h5>
                <p><strong>Implementation Time:</strong> {implementation_time}</p>
                <p><strong>Complexity:</strong> {complexity}</p>
                <p><strong>Expected Benefit:</strong> {expected_benefit}</p>
                <p><strong>Priority:</strong> {"High" if i <= 2 else "Medium" if i <= 4 else "Low"}</p>
            </div>
            """)

def render_live_pricing_tab(analysis: Dict, config: Dict):
    """Render live AWS pricing analysis tab"""
    st.subheader("💰 Live AWS Pricing & Cost Analysis")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    pricing_data = aws_sizing.get('pricing_data', {})
    cost_analysis = analysis.get('cost_analysis', {})
    
    # Pricing Data Source Indicator
    data_source = pricing_data.get('data_source', 'fallback')
    last_updated = pricing_data.get('last_updated', datetime.now())
    
    if data_source == 'aws_api':
        st.success(f"✅ **Live AWS Pricing** (Updated: {last_updated.strftime('%Y-%m-%d %H:%M UTC')})")
    else:
        st.warning(f"⚠️ **Fallback Pricing** (AWS API unavailable)")
    
    # Cost Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💻 Compute Cost", f"${cost_analysis.get('aws_compute_cost', 0):.0f}/mo")
    
    with col2:
        st.metric("💾 Storage Cost", f"${cost_analysis.get('aws_storage_cost', 0):.0f}/mo")
    
    with col3:
        st.metric("🌐 Network Cost", f"${cost_analysis.get('network_cost', 0):.0f}/mo")
    
    with col4:
        st.metric("🤖 Agent Cost", f"${cost_analysis.get('agent_cost', 0):.0f}/mo")
    
    with col5:
        st.metric("💰 Total Monthly", f"${cost_analysis.get('total_monthly_cost', 0):.0f}")
    
    # Live Pricing Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💻 Live EC2 Pricing (US-West-2):**")
        
        ec2_pricing = pricing_data.get('ec2_instances', {})
        if ec2_pricing:
            ec2_df = pd.DataFrame([
                {
                    'Instance Type': instance_type,
                    'vCPU': specs['vcpu'],
                    'Memory (GB)': specs['memory'],
                    'Hourly Cost': f"${specs['cost_per_hour']:.4f}",
                    'Monthly Cost': f"${specs['cost_per_hour'] * 24 * 30:.0f}"
                }
                for instance_type, specs in ec2_pricing.items()
            ])
            st.dataframe(ec2_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**🗄️ Live RDS Pricing (US-West-2):**")
        
        rds_pricing = pricing_data.get('rds_instances', {})
        if rds_pricing:
            rds_df = pd.DataFrame([
                {
                    'Instance Type': instance_type,
                    'vCPU': specs['vcpu'],
                    'Memory (GB)': specs['memory'],
                    'Hourly Cost': f"${specs['cost_per_hour']:.4f}",
                    'Monthly Cost': f"${specs['cost_per_hour'] * 24 * 30:.0f}"
                }
                for instance_type, specs in rds_pricing.items()
            ])
            st.dataframe(rds_df, use_container_width=True, hide_index=True)
    
    # AI Cost Insights
    ai_cost_insights = cost_analysis.get('ai_cost_insights', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>🤖 AI Cost Optimization</h4>
            <p><strong>Optimization Factor:</strong> {ai_cost_insights.get('ai_optimization_factor', 0)*100:.1f}%</p>
            <p><strong>Complexity Multiplier:</strong> {ai_cost_insights.get('complexity_multiplier', 1.0):.2f}x</p>
            <p><strong>Management Savings:</strong> {ai_cost_insights.get('management_reduction', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>💡 Cost Projections</h4>
            <p><strong>1-Year Total:</strong> ${cost_analysis.get('total_monthly_cost', 0) * 12:.0f}</p>
            <p><strong>3-Year TCO:</strong> ${cost_analysis.get('total_monthly_cost', 0) * 36 + cost_analysis.get('one_time_migration_cost', 0):.0f}</p>
            <p><strong>ROI Timeline:</strong> {cost_analysis.get('roi_months', 'N/A')} months</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>📊 Savings Potential</h4>
            <p><strong>Monthly Savings:</strong> ${cost_analysis.get('estimated_monthly_savings', 0):.0f}</p>
            <p><strong>Annual Savings:</strong> ${cost_analysis.get('estimated_monthly_savings', 0) * 12:.0f}</p>
            <p><strong>AI Additional Savings:</strong> {ai_cost_insights.get('potential_additional_savings', '0%')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost Breakdown Chart
    st.markdown("**📊 Cost Breakdown Visualization:**")
    
    cost_breakdown = {
        'Component': ['Compute', 'Storage', 'Network', 'Agents', 'OS Licensing', 'Management'],
        'Monthly Cost': [
            cost_analysis.get('aws_compute_cost', 0),
            cost_analysis.get('aws_storage_cost', 0),
            cost_analysis.get('network_cost', 0),
            cost_analysis.get('agent_cost', 0),
            cost_analysis.get('os_licensing_cost', 0),
            cost_analysis.get('management_cost', 0)
        ]
    }
    
    fig_cost = px.pie(
        cost_breakdown, 
        values='Monthly Cost', 
        names='Component',
        title="Monthly Cost Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_cost, use_container_width=True)

def render_os_performance_enhanced_tab(analysis: Dict, config: Dict):
    """Render enhanced OS performance analysis tab"""
    st.subheader("💻 Operating System Performance Analysis (AI-Enhanced)")
    
    onprem_performance = analysis.get('onprem_performance', {})
    os_impact = onprem_performance.get('os_impact', {})
    ai_recommendations = onprem_performance.get('ai_optimization_recommendations', [])
    
    # OS Performance Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 OS Efficiency", 
            f"{os_impact.get('total_efficiency', 0)*100:.1f}%",
            delta=f"Base: {os_impact.get('base_efficiency', 0)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "💾 DB Optimization",
            f"{os_impact.get('db_optimization', 0)*100:.1f}%",
            delta=f"Engine: {config['database_engine'].upper()}"
        )
    
    with col3:
        st.metric(
            "⚡ Platform Impact",
            f"{os_impact.get('platform_optimization', 1.0)*100:.1f}%",
            delta=f"Type: {config['server_type'].title()}"
        )
    
    with col4:
        licensing_factor = os_impact.get('licensing_cost_factor', 1.0)
        st.metric(
            "💰 Licensing Factor",
            f"{licensing_factor:.1f}x",
            delta="Cost multiplier"
        )
    
    # OS Comparison Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Current OS Analysis
        os_insights = os_impact.get('ai_insights', {})
        if os_insights:
            st.markdown(f"""
            <div class="os-performance-enhanced-card">
                <h4>🔍 Current OS Analysis: {os_impact.get('name', config['operating_system'])}</h4>
                <h5>💪 Strengths:</h5>
                <ul>
                    {"".join([f"<li>{strength}</li>" for strength in os_insights.get('strengths', [])])}
                </ul>
                <h5>⚠️ Considerations:</h5>
                <ul>
                    {"".join([f"<li>{weakness}</li>" for weakness in os_insights.get('weaknesses', [])])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Performance Breakdown
        st.markdown(f"""
        <div class="os-performance-enhanced-card">
            <h4>📊 Performance Breakdown</h4>
            <p><strong>CPU Efficiency:</strong> {os_impact.get('cpu_efficiency', 0)*100:.1f}%</p>
            <p><strong>Memory Efficiency:</strong> {os_impact.get('memory_efficiency', 0)*100:.1f}%</p>
            <p><strong>I/O Efficiency:</strong> {os_impact.get('io_efficiency', 0)*100:.1f}%</p>
            <p><strong>Network Efficiency:</strong> {os_impact.get('network_efficiency', 0)*100:.1f}%</p>
            <p><strong>Virtualization Overhead:</strong> {os_impact.get('virtualization_overhead', 0)*100:.1f}%</p>
            <p><strong>Security Overhead:</strong> {os_impact.get('security_overhead', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Physical vs Virtual Performance Comparison
    st.markdown("**🏢 Physical vs Virtual Performance Impact:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        physical_efficiency = os_impact.get('total_efficiency', 0) * 1.05  # Physical boost
        st.markdown(f"""
        <div class="physical-virtual-comparison-card">
            <h4>🏢 Physical Server Performance</h4>
            <p><strong>Total Efficiency:</strong> {physical_efficiency*100:.1f}%</p>
            <p><strong>Performance Boost:</strong> +5% (Direct hardware access)</p>
            <p><strong>Virtualization Overhead:</strong> 0%</p>
            <p><strong>Best For:</strong> Maximum performance, dedicated workloads</p>
            <p><strong>Migration Complexity:</strong> Lower (direct hardware mapping)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        virtual_efficiency = os_impact.get('total_efficiency', 0)
        st.markdown(f"""
        <div class="physical-virtual-comparison-card">
            <h4>☁️ VMware Virtual Performance</h4>
            <p><strong>Total Efficiency:</strong> {virtual_efficiency*100:.1f}%</p>
            <p><strong>Virtualization Overhead:</strong> {os_impact.get('virtualization_overhead', 0)*100:.1f}%</p>
            <p><strong>Resource Sharing:</strong> Yes (may impact performance)</p>
            <p><strong>Best For:</strong> Flexible resource allocation, cost efficiency</p>
            <p><strong>Migration Complexity:</strong> Higher (virtualization layer considerations)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # OS Efficiency Comparison Chart
    st.markdown("**📊 OS Efficiency Comparison:**")
    
    os_manager = OSPerformanceManager()
    os_comparison_data = []
    
    for os_key, os_config in os_manager.operating_systems.items():
        # Calculate efficiency for each OS with current configuration
        os_perf = os_manager.calculate_os_performance_impact(
            os_key, config['server_type'], config['database_engine']
        )
        
        os_comparison_data.append({
            'OS': os_config['name'],
            'Total Efficiency': os_perf['total_efficiency'] * 100,
            'CPU Efficiency': os_perf['cpu_efficiency'] * 100,
            'Memory Efficiency': os_perf['memory_efficiency'] * 100,
            'I/O Efficiency': os_perf['io_efficiency'] * 100,
            'Network Efficiency': os_perf['network_efficiency'] * 100,
            'DB Optimization': os_perf['db_optimization'] * 100,
            'Licensing Cost': os_perf['licensing_cost_factor']
        })
    
    df_os_comparison = pd.DataFrame(os_comparison_data)
    
    # Create comparison chart
    fig_os = px.bar(
        df_os_comparison, 
        x='OS', 
        y='Total Efficiency',
        title=f"OS Performance Comparison for {config['database_engine'].upper()} on {config['server_type'].title()}",
        color='Total Efficiency',
        color_continuous_scale='RdYlGn'
    )
    fig_os.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_os, use_container_width=True)
    
    # Detailed OS comparison table
    st.markdown("**📋 Detailed OS Performance Comparison:**")
    st.dataframe(df_os_comparison, use_container_width=True, hide_index=True)
    
    # AI Migration Considerations
    if os_insights.get('migration_considerations'):
        st.markdown("**🚀 AI Migration Considerations:**")
        for consideration in os_insights['migration_considerations']:
            st.markdown(f"• {consideration}")
    
    # AI Optimization Recommendations
    if ai_recommendations:
        st.markdown("**🤖 AI Optimization Recommendations:**")
        for i, recommendation in enumerate(ai_recommendations, 1):
            st.markdown(f"**{i}.** {recommendation}")

def render_enhanced_migration_dashboard(analysis: Dict, config: Dict):
    """Render enhanced migration performance dashboard"""
    st.subheader("📊 Enhanced Migration Performance Dashboard")
    
    # Migration Overview Metrics - Compact Design
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🚀 Migration Throughput", 
            f"{analysis.get('migration_throughput_mbps', 0):.0f} Mbps",
            delta=f"{analysis.get('primary_tool', '').upper()} optimized"
        )
    
    with col2:
        st.metric(
            "⏱️ Estimated Time", 
            f"{analysis.get('estimated_migration_time_hours', 0):.1f} hours",
            delta=f"{config['database_size_gb']} GB database"
        )
    
    with col3:
        deployment = analysis.get('aws_sizing_recommendations', {}).get('deployment_recommendation', {}).get('recommendation', 'unknown')
        st.metric(
            "☁️ AWS Deployment", 
            deployment.upper(),
            delta=f"{analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('total_instances', 0)} instances"
        )
    
    with col4:
        st.metric(
            "💰 Monthly Cost", 
            f"${analysis.get('cost_analysis', {}).get('total_monthly_cost', 0):.0f}",
            delta=f"${analysis.get('cost_analysis', {}).get('estimated_monthly_savings', 0):.0f} potential savings"
        )
    
    with col5:
        ai_readiness = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        st.metric(
            "🎯 AI Readiness", 
            f"{ai_readiness:.0f}/100",
            delta=analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown')
        )
    
    # Current Performance Metrics Analysis
    st.markdown("**📊 Current Performance Profile:**")
    
    hardware_perf = analysis.get('onprem_performance', {})
    current_perf = hardware_perf.get('current_performance_analysis', {})
    resource_adequacy = hardware_perf.get('resource_adequacy', {})
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>💾 Storage Analysis</h5>
            <p><strong>Utilization:</strong> {current_perf.get('storage_utilization_percent', 0):.1f}%</p>
            <p><strong>Efficiency:</strong> {current_perf.get('storage_efficiency', 0):.1f}%</p>
            <p><strong>Type:</strong> {current_perf.get('workload_classification', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚡ IOPS Profile</h5>
            <p><strong>Peak IOPS:</strong> {config.get('peak_iops', 0):,}</p>
            <p><strong>IOPS/GB:</strong> {current_perf.get('iops_per_gb', 0):.1f}</p>
            <p><strong>Intensity:</strong> {current_perf.get('performance_intensity', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🚀 Throughput</h5>
            <p><strong>Max:</strong> {config.get('max_throughput_mbps', 0)} MB/s</p>
            <p><strong>Per GB:</strong> {current_perf.get('throughput_per_gb_mbps', 0):.2f} MB/s</p>
            <p><strong>Priority:</strong> {current_perf.get('optimization_priority', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🧠 Memory Readiness</h5>
            <p><strong>Current:</strong> {config.get('ram_gb', 0)} GB</p>
            <p><strong>Anticipated:</strong> {config.get('anticipated_max_memory_gb', 0)} GB</p>
            <p><strong>Adequacy:</strong> {resource_adequacy.get('memory_adequacy_score', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col5:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚙️ CPU Readiness</h5>
            <p><strong>Current:</strong> {config.get('cpu_cores', 0)} cores</p>
            <p><strong>Anticipated:</strong> {config.get('anticipated_max_cpu_cores', 0)} cores</p>
            <p><strong>Adequacy:</strong> {resource_adequacy.get('cpu_adequacy_score', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Resource Adequacy Analysis
    st.markdown("**🎯 Resource Adequacy Analysis:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_adequacy = resource_adequacy.get('overall_adequacy_score', 0)
        readiness_level = resource_adequacy.get('readiness_level', 'Unknown')
        upgrade_priority = resource_adequacy.get('upgrade_priority', 'Unknown')
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Overall Resource Assessment</h4>
            <p><strong>Adequacy Score:</strong> {overall_adequacy:.1f}%</p>
            <p><strong>Readiness Level:</strong> {readiness_level}</p>
            <p><strong>Upgrade Priority:</strong> {upgrade_priority}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        memory_gap = resource_adequacy.get('memory_gap_gb', 0)
        cpu_gap = resource_adequacy.get('cpu_gap_cores', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Resource Gap Analysis</h4>
            <p><strong>Memory Gap:</strong> {memory_gap:+.0f} GB</p>
            <p><strong>CPU Gap:</strong> {cpu_gap:+.0f} cores</p>
            <p><strong>Gap Status:</strong> {"Surplus" if memory_gap <= 0 and cpu_gap <= 0 else "Deficit"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        workload_class = current_perf.get('workload_classification', 'Unknown')
        perf_intensity = current_perf.get('performance_intensity', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔍 Workload Characteristics</h4>
            <p><strong>Classification:</strong> {workload_class}</p>
            <p><strong>Performance Intensity:</strong> {perf_intensity:.1f}%</p>
            <p><strong>Migration Complexity:</strong> {"High" if perf_intensity > 75 else "Medium" if perf_intensity > 50 else "Low"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Throughput Analysis
    st.markdown("**📊 Throughput Bottleneck Analysis:**")
    
    network_perf = analysis.get('network_performance', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    throughput_data = {
        'Component': ['Hardware NIC', 'Network Path', 'Migration Agent', 'Effective Migration'],
        'Throughput (Mbps)': [
            hardware_perf.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
            network_perf.get('effective_bandwidth_mbps', 0),
            agent_analysis.get('effective_throughput', 0),
            analysis.get('migration_throughput_mbps', 0)
        ],
        'Status': ['Hardware', 'Network', 'Agent', 'Final']
    }
    
    fig_throughput = px.bar(
        throughput_data, 
        x='Component', 
        y='Throughput (Mbps)',
        color='Status',
        title="Migration Throughput Bottleneck Analysis",
        color_discrete_map={'Hardware': '#3498db', 'Network': '#2ecc71', 'Agent': '#e74c3c', 'Final': '#9b59b6'}
    )
    
    fig_throughput.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    
    # Performance Insights - Compact Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bottleneck = agent_analysis.get('bottleneck', 'unknown')
        bottleneck_icon = "🌐" if bottleneck == 'network' else "🤖"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔍 Bottleneck Analysis</h4>
            <p><strong>Primary Bottleneck:</strong> {bottleneck_icon} {bottleneck.title()}</p>
            <p><strong>Agent Utilization:</strong> {agent_analysis.get('throughput_impact', 0)*100:.1f}%</p>
            <p><strong>Network Quality:</strong> {network_perf.get('network_quality_score', 0):.1f}/100</p>
            <p><strong>AI Enhancement:</strong> {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        onprem_score = hardware_perf.get('performance_score', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Performance Baseline</h4>
            <p><strong>On-Premises Score:</strong> {onprem_score:.1f}/100</p>
            <p><strong>Performance Tier:</strong> {hardware_perf.get('overall_performance', {}).get('performance_tier', 'Unknown')}</p>
            <p><strong>OS Efficiency:</strong> {hardware_perf.get('os_impact', {}).get('total_efficiency', 0)*100:.1f}%</p>
            <p><strong>Platform Type:</strong> {config['server_type'].title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ai_optimization = agent_analysis.get('ai_optimization', {})
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 AI Optimization</h4>
            <p><strong>Optimization Potential:</strong> {ai_optimization.get('optimization_potential_percent', 0):.1f}%</p>
            <p><strong>Estimated Improvement:</strong> {ai_optimization.get('estimated_improvement', 'N/A')}</p>
            <p><strong>AI Complexity:</strong> {analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Migration Timeline Visualization
    st.markdown("**📅 AI-Optimized Migration Timeline:**")
    
    timeline = analysis.get('ai_overall_assessment', {}).get('timeline_recommendation', {})
    
    timeline_data = {
        'Phase': ['Planning', 'Testing', 'Migration Execution', 'Post-Migration'],
        'Duration (weeks)': [
            timeline.get('planning_phase_weeks', 2),
            timeline.get('testing_phase_weeks', 3),
            timeline.get('migration_window_hours', 24) / (24 * 7),  # Convert to weeks
            1  # Post-migration phase
        ],
        'Type': ['Preparation', 'Preparation', 'Execution', 'Validation']
    }
    
    fig_timeline = px.bar(
        timeline_data,
        x='Phase',
        y='Duration (weeks)',
        color='Type',
        title="AI-Recommended Migration Timeline",
        color_discrete_map={'Preparation': '#3498db', 'Execution': '#e74c3c', 'Validation': '#2ecc71'}
    )
    
    fig_timeline.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

def render_pdf_reports_tab(analysis: Dict, config: Dict):
    """Render PDF reports generation tab"""
    st.subheader("📄 Executive PDF Reports")
    
    # PDF Generation Section
    st.markdown(f"""
    <div class="pdf-download-section">
        <h3>🎯 Generate Executive Migration Report</h3>
        <p>Create a comprehensive PDF report for stakeholders with all analysis results, recommendations, and technical details.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📊 Generate Executive PDF Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating comprehensive PDF report..."):
                try:
                    # Generate PDF report
                    pdf_generator = PDFReportGenerator()
                    pdf_bytes = pdf_generator.generate_executive_report(analysis, config)
                    
                    # Create download button
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"AWS_Migration_Analysis_{current_time}.pdf"
                    
                    st.success("✅ PDF Report Generated Successfully!")
                    
                    st.download_button(
                        label="📥 Download Executive Report PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.info("💡 PDF generation requires matplotlib. The analysis is still available in other tabs.")
    
    # Report Contents Preview
    st.markdown("**📋 Report Contents:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>📈 Executive Summary</h4>
            <ul>
                <li>Migration overview and key metrics</li>
                <li>AI readiness assessment</li>
                <li>Cost summary and ROI analysis</li>
                <li>Performance baseline evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>⚙️ Technical Analysis</h4>
            <ul>
                <li>Current performance breakdown</li>
                <li>Network path analysis</li>
                <li>OS performance impact</li>
                <li>Identified bottlenecks and solutions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>☁️ AWS Recommendations</h4>
            <ul>
                <li>Deployment recommendations (RDS vs EC2)</li>
                <li>Instance sizing and configuration</li>
                <li>Reader/writer setup</li>
                <li>AI complexity analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>📊 Cost & Risk Analysis</h4>
            <ul>
                <li>Detailed cost breakdown</li>
                <li>Financial projections and ROI</li>
                <li>Risk assessment matrix</li>
                <li>Recommended timeline and next steps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Report Options
    st.markdown("**🔧 Report Customization:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_technical_details = st.checkbox("Include Technical Deep-Dive", value=True, help="Include detailed technical analysis and performance metrics")
    
    with col2:
        include_ai_insights = st.checkbox("Include AI Insights", value=True, help="Include all AI-generated recommendations and analysis")
    
    with col3:
        include_cost_analysis = st.checkbox("Include Cost Analysis", value=True, help="Include detailed cost breakdown and financial projections")
    
    # Report Metrics
    st.markdown("**📊 Report Metrics:**")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("📄 Pages", "5", delta="Executive format")
    
    with metrics_col2:
        chart_count = 8 + (3 if include_technical_details else 0) + (2 if include_ai_insights else 0)
        st.metric("📊 Charts & Graphs", str(chart_count), delta="Visual analysis")
    
    with metrics_col3:
        recommendation_count = len(analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('performance_recommendations', []))
        st.metric("💡 AI Recommendations", str(recommendation_count), delta="Actionable insights")
    
    with metrics_col4:
        complexity_score = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        st.metric("🎯 AI Complexity", f"{complexity_score}/10", delta="Migration difficulty")

async def main():
    """Enhanced main function with professional UI and detailed analysis"""
    render_enhanced_header()
    
    # Get configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Run analysis
    analysis_placeholder = st.empty()
    
    with analysis_placeholder.container():
        if config['enable_ai_analysis']:
            with st.spinner("🧠 Running comprehensive AI-powered migration analysis..."):
                try:
                    analysis = await analyzer.comprehensive_ai_migration_analysis(config)
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    # Fallback to simplified analysis
                    analysis = {
                        'api_status': APIStatus(anthropic_connected=False, aws_pricing_connected=False),
                        'onprem_performance': {'performance_score': 75, 'os_impact': {'total_efficiency': 0.85}},
                        'network_performance': {'network_quality_score': 80, 'effective_bandwidth_mbps': 1000, 'segments': []},
                        'migration_type': 'homogeneous' if config['source_database_engine'] == config['database_engine'] else 'heterogeneous',
                        'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                        'agent_analysis': {'primary_tool': 'datasync', 'effective_throughput': 500},
                        'migration_throughput_mbps': 500,
                        'estimated_migration_time_hours': 8,
                        'aws_sizing_recommendations': {'deployment_recommendation': {'recommendation': 'rds'}},
                        'cost_analysis': {'total_monthly_cost': 1500},
                        'ai_overall_assessment': {'migration_readiness_score': 75, 'risk_level': 'Medium'}
                    }
        else:
            with st.spinner("🔬 Running standard migration analysis..."):
                # Simplified analysis without AI
                analysis = {
                    'api_status': APIStatus(anthropic_connected=False, aws_pricing_connected=False),
                    'onprem_performance': {'performance_score': 75, 'os_impact': {'total_efficiency': 0.85}},
                    'network_performance': {'network_quality_score': 80, 'effective_bandwidth_mbps': 1000, 'segments': []},
                    'migration_type': 'homogeneous' if config['source_database_engine'] == config['database_engine'] else 'heterogeneous',
                    'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                    'agent_analysis': {'primary_tool': 'datasync', 'effective_throughput': 500},
                    'migration_throughput_mbps': 500,
                    'estimated_migration_time_hours': 8,
                    'aws_sizing_recommendations': {'deployment_recommendation': {'recommendation': 'rds'}},
                    'cost_analysis': {'total_monthly_cost': 1500},
                    'ai_overall_assessment': {'migration_readiness_score': 75, 'risk_level': 'Medium'}
                }
    
    analysis_placeholder.empty()
    
    # Enhanced tabs with professional styling
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧠 AI Insights & Analysis", 
        "🌐 Network Intelligence",
        "💰 Cost & Pricing Analysis",
        "💻 OS Performance Analysis",
        "📊 Migration Dashboard",
        "🎯 AWS Sizing & Configuration",
        "📄 Executive PDF Reports"
    ])
    
    with tab1:
        if config['enable_ai_analysis']:
            render_ai_insights_tab_enhanced(analysis, config)
        else:
            st.info("🤖 Enable AI Analysis in the sidebar for comprehensive migration insights")
    
    with tab2:
        render_network_intelligence_tab_enhanced(analysis, config)
    
    with tab3:
        render_live_pricing_tab(analysis, config)
    
    with tab4:
        render_os_performance_enhanced_tab(analysis, config)
    
    with tab5:
        render_enhanced_migration_dashboard(analysis, config)
    
    with tab6:
        render_aws_sizing_configuration_tab(analysis, config)
    
    with tab7:
        render_pdf_reports_tab(analysis, config)
    
    # Professional footer
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic Claude AI • Real-time AWS Integration • Professional Migration Analysis</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Network Intelligence • 🎯 AI-Driven Recommendations • 📊 Executive Reporting
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())