from flask import Flask, jsonify, render_template
from flask_cors import CORS
import deployment.model as model
import os
import traceback

app = Flask(__name__)
CORS(app)  

@app.route('/')
def index():
    """Serve the HTML page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    print("Inside get_recommendations function", user_id)
    """Get recommendations for a user by calling model.py function"""
    try:
        # Call the function from model.py with default weights (0.6 CF, 0.4 Sentiment)
        product_names = model.recommend_products_hybrid(user_id, cf_weight=0.4, sentiment_weight=0.6)
        print("Product names: ", product_names)
        
        # Handle None values in product names
        if not product_names or len(product_names) == 0:
            return jsonify({
                'user_id': user_id,
                'error': 'No recommendations found for this user',
                'recommendations': []
            }), 200
        
        recommendations = [
            {'product_name': name if name else 'Unknown', 'rank': idx + 1}
            for idx, name in enumerate(product_names)
            if name is not None
        ]
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        })
    
    except KeyError:
        return jsonify({
            'error': f'User "{user_id}" not found',
            'available_users': list(model.user_final_rating.index[:10])
        }), 404
    
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)