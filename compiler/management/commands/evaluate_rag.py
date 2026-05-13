"""
Django Management Command for RAG Evaluation
Usage: python manage.py evaluate_rag
"""

from django.core.management.base import BaseCommand, CommandError
from rag.rag_service import RAGService
import json

class Command(BaseCommand):
    help = 'Evaluate RAG retrieval performance using Recall@k, Precision@k, and MRR metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--k-values',
            type=str,
            default='1,3,5',
            help='Comma-separated k values for evaluation (default: 1,3,5)'
        )
        
        parser.add_argument(
            '--save-results',
            action='store_true',
            help='Save evaluation results to file'
        )
        
        parser.add_argument(
            '--output-file',
            type=str,
            help='Output file path for results (optional)'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

    def handle(self, *args, **options):
        """Execute the RAG evaluation command"""
        
        # Parse k values
        try:
            k_values = [int(k.strip()) for k in options['k_values'].split(',')]
        except ValueError:
            raise CommandError('Invalid k-values format. Use comma-separated integers.')
        
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting RAG Evaluation Pipeline')
        )
        self.stdout.write('=' * 50)
        
        try:
            # Initialize RAG service
            rag_service = RAGService()
            
            if not rag_service.is_available():
                raise CommandError('RAG service is not available. Please ensure vector store is loaded.')
            
            self.stdout.write(
                self.style.SUCCESS('✅ RAG service initialized successfully')
            )
            
            # Run evaluation
            if options['verbose']:
                self.stdout.write(f"📊 Evaluating with k values: {k_values}")
            
            eval_results = rag_service.evaluate_retrieval_performance(k_values=k_values)
            
            if 'error' in eval_results:
                raise CommandError(f"Evaluation failed: {eval_results['error']}")
            
            # Display results
            self.stdout.write('\n' + '=' * 50)
            self.stdout.write(
                self.style.SUCCESS('📈 EVALUATION RESULTS')
            )
            self.stdout.write('=' * 50)
            
            num_queries = eval_results.get('num_queries', 0)
            mrr_score = eval_results['metrics']['mrr']['score']
            
            self.stdout.write(f"📊 Number of queries evaluated: {num_queries}")
            self.stdout.write(f"📊 MRR Score: {mrr_score:.4f}")
            
            self.stdout.write('\n📊 Recall@k and Precision@k:')
            
            for k in k_values:
                if f'recall@{k}' in eval_results['metrics']:
                    recall_data = eval_results['metrics'][f'recall@{k}']
                    precision_data = eval_results['metrics'][f'precision@{k}']
                    
                    recall_mean = recall_data['mean']
                    recall_std = recall_data['std']
                    precision_mean = precision_data['mean']
                    precision_std = precision_data['std']
                    
                    self.stdout.write(
                        f"  k={k}: Recall={recall_mean:.4f}±{recall_std:.4f}, "
                        f"Precision={precision_mean:.4f}±{precision_std:.4f}"
                    )
            
            # Save results if requested
            if options['save_results']:
                if options['output_file']:
                    filepath = rag_service.evaluator.save_evaluation_results(
                        eval_results, options['output_file']
                    )
                else:
                    filepath = rag_service.evaluator.save_evaluation_results(eval_results)
                
                self.stdout.write(f"\n💾 Results saved to: {filepath}")
            
            # Show detailed metrics if verbose
            if options['verbose']:
                self.stdout.write('\n📊 Detailed Metrics:')
                for metric_name, metric_data in eval_results['metrics'].items():
                    if 'scores' in metric_data:
                        scores = metric_data['scores']
                        self.stdout.write(f"  {metric_name}: {scores}")
            
            self.stdout.write(
                self.style.SUCCESS('\n✅ Evaluation completed successfully!')
            )
            
        except Exception as e:
            raise CommandError(f'Evaluation failed: {str(e)}')
