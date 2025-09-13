# app.py
# Flask web application for CAMS air pollution visualization

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import traceback
import calendar

# Import our custom modules
from data_processor import NetCDFProcessor, analyze_netcdf_file
from plot_generator import IndiaMapPlotter
from cams_downloader import CAMSDownloader
from constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, COLOR_THEMES

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this!
app.config['DEBUG'] = False  # Explicitly disable debug mode

# Add JSON filter for templates
import json
app.jinja_env.filters['tojson'] = json.dumps

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize our services
downloader = CAMSDownloader()
plotter = IndiaMapPlotter()

# Ensure directories exist
for directory in ['uploads', 'downloads', 'plots', 'templates', 'static']:
    Path(directory).mkdir(exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def str_to_bool(value):
    """Convert string representation to boolean"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


@app.route('/')
def index():
    """Main page - file upload or date selection"""
    downloaded_files = downloader.list_downloaded_files()
    
    # Pass the current date to the template for the date picker's max attribute
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                         downloaded_files=downloaded_files,
                         cds_ready=downloader.is_client_ready(),
                         current_date=current_date)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            
            file.save(str(filepath))
            flash(f'File uploaded successfully: {filename}', 'success')
            
            return redirect(url_for('analyze_file', filename=filename))
            
        except Exception as e:
            flash(f'Error uploading file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload .nc or .zip files.', 'error')
        return redirect(url_for('index'))


@app.route('/download_date', methods=['POST'])
def download_date():
    """Handle date-based download"""
    date_str = request.form.get('date')
    
    if not date_str:
        flash('Please select a date', 'error')
        return redirect(url_for('index'))
    
    # --- Backend Validation Logic ---
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d')
        start_date = datetime(2015, 1, 1)
        end_date = datetime.now()
        
        if not (start_date <= selected_date <= end_date):
            flash(f'Invalid date. Please select a date between {start_date.strftime("%Y-%m-%d")} and today.', 'error')
            return redirect(url_for('index'))
            
    except ValueError:
        flash('Invalid date format. Please use YYYY-MM-DD.', 'error')
        return redirect(url_for('index'))
    
    # --- End of Validation Logic ---
    
    if not downloader.is_client_ready():
        flash('CDS API not configured. Please check your .cdsapirc file.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Download CAMS data
        zip_path = downloader.download_cams_data(date_str)
        
        # Extract the files
        extracted_files = downloader.extract_cams_files(zip_path)
        
        flash(f'CAMS data downloaded successfully for {date_str}', 'success')
        
        # Analyze the extracted files
        if 'surface' in extracted_files:
            filename = Path(extracted_files['surface']).name
            return redirect(url_for('analyze_file', filename=filename, is_download='true'))
        elif 'atmospheric' in extracted_files:
            filename = Path(extracted_files['atmospheric']).name
            return redirect(url_for('analyze_file', filename=filename, is_download='true'))
        else:
            # Use the first available file
            first_file = list(extracted_files.values())[0]
            filename = Path(first_file).name
            return redirect(url_for('analyze_file', filename=filename, is_download='true'))
            
    except Exception as e:
        flash(f'Error downloading CAMS data: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/analyze/<filename>')
def analyze_file(filename):
    """Analyze uploaded file and show variable selection"""
    is_download_param = request.args.get('is_download', 'false')
    is_download = str_to_bool(is_download_param)
    
    try:
        # Determine file path
        if is_download:
            file_path = Path('downloads/extracted') / filename
        else:
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        
        if not file_path.exists():
            flash('File not found', 'error')
            return redirect(url_for('index'))
        
        # Analyze the file
        analysis = analyze_netcdf_file(str(file_path))
        
        if not analysis['success']:
            flash(f'Error analyzing file: {analysis["error"]}', 'error')
            return redirect(url_for('index'))
        
        if analysis['total_variables'] == 0:
            flash('No air pollution variables found in the file', 'warning')
            return redirect(url_for('index'))
        
        # Process variables for template
        variables = []
        for var_name, var_info in analysis['detected_variables'].items():
            variables.append({
                'name': var_name,
                'display_name': var_info['name'],
                'type': var_info['type'],
                'units': var_info['units'],
                'shape': var_info['shape']
            })
        
        return render_template('variables.html',
                             filename=filename,
                             variables=variables,
                             color_themes=COLOR_THEMES,
                             is_download=is_download)
        
    except Exception as e:
        flash(f'Error analyzing file: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/get_pressure_levels/<filename>/<variable>')
def get_pressure_levels(filename, variable):
    """AJAX endpoint to get pressure levels for atmospheric variables"""
    try:
        is_download_param = request.args.get('is_download', 'false')
        is_download = str_to_bool(is_download_param)
        
        print(f"is_download: {is_download} (type: {type(is_download)})")

        # Determine file path
        if is_download:
            file_path = Path('downloads/extracted') / filename
            print("Using downloaded file path")
        else:
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
            print("Using upload file path")
        
        print(f"File path: {file_path}")

        processor = NetCDFProcessor(str(file_path))
        processor.load_dataset()
        processor.detect_variables()
        
        pressure_levels = processor.get_available_pressure_levels(variable)
        processor.close()
        
        return jsonify({
            'success': True,
            'pressure_levels': pressure_levels
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/get_available_times/<filename>/<variable>')
def get_available_times(filename, variable):
    """AJAX endpoint to get available timestamps for a variable"""
    try:
        is_download_param = request.args.get('is_download', 'false')
        is_download = str_to_bool(is_download_param)
        
        # Determine file path
        if is_download:
            file_path = Path('downloads/extracted') / filename
        else:
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        
        processor = NetCDFProcessor(str(file_path))
        processor.load_dataset()
        processor.detect_variables()
        
        available_times = processor.get_available_times(variable)
        processor.close()
        
        # Format times for display
        formatted_times = []
        for i, time_val in enumerate(available_times):
            formatted_times.append({
                'index': i,
                'value': str(time_val),
                'display': time_val.strftime('%Y-%m-%d %H:%M') if hasattr(time_val, 'strftime') else str(time_val)
            })
        
        return jsonify({
            'success': True,
            'times': formatted_times
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/visualize', methods=['POST'])
def visualize():
    """Generate and display the pollution map"""
    try:
        filename = request.form.get('filename')
        variable = request.form.get('variable')
        color_theme = request.form.get('color_theme', 'viridis')
        pressure_level = request.form.get('pressure_level')
        is_download_param = request.form.get('is_download', 'false')
        is_download = str_to_bool(is_download_param)
        
        if not filename or not variable:
            flash('Missing required parameters', 'error')
            return redirect(url_for('index'))
        
        # Determine file path
        if is_download:
            file_path = Path('downloads/extracted') / filename
        else:
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        
        if not file_path.exists():
            flash('File not found', 'error')
            return redirect(url_for('index'))
        
        # Process the data
        processor = NetCDFProcessor(str(file_path))
        processor.load_dataset()
        processor.detect_variables()
        
        # Convert pressure level to float if provided
        pressure_level_val = None
        if pressure_level and pressure_level != 'None':
            try:
                pressure_level_val = float(pressure_level)
            except ValueError:
                pressure_level_val = None

        time_index_val = request.form.get('time_index')
        # Extract data
        data_values, metadata = processor.extract_data(
            variable, 
            time_index = int(time_index_val) if time_index_val and time_index_val != 'None' else 0,
            pressure_level=pressure_level_val
        )
        
        # Generate plot
        plot_path = plotter.create_india_map(
            data_values, 
            metadata, 
            color_theme=color_theme,
            save_plot=True
        )
        
        processor.close()
        
        if plot_path:
            plot_filename = Path(plot_path).name
            
            # Prepare metadata for display
            plot_info = {
                'variable': metadata.get('display_name', 'Unknown Variable'),
                'units': metadata.get('units', ''),
                'shape': str(metadata.get('shape', 'Unknown')),
                'pressure_level': metadata.get('pressure_level'),
                'color_theme': COLOR_THEMES.get(color_theme, color_theme),
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_range': {
                    'min': float(f"{data_values.min():.3f}") if hasattr(data_values, 'min') and not data_values.min() is None else 0,
                    'max': float(f"{data_values.max():.3f}") if hasattr(data_values, 'max') and not data_values.max() is None else 0,
                    'mean': float(f"{data_values.mean():.3f}") if hasattr(data_values, 'mean') and not data_values.mean() is None else 0
                }
            }
            
            print(f"Plot info prepared: {plot_info}")
            
            return render_template('plot.html',
                                    plot_filename=plot_filename,
                                    plot_info=plot_info)
        else:
            flash('Error generating plot', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error creating visualization: {str(e)}', 'error')
        print(f"Full error: {traceback.format_exc()}")
        return redirect(url_for('index'))

@app.route('/plot/<filename>')
def serve_plot(filename):
    """Serve plot images"""
    try:
        plot_path = Path('plots') / filename
        if plot_path.exists():
            return send_file(str(plot_path), mimetype='image/png')
        else:
            flash('Plot not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error serving plot: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/cleanup')
def cleanup():
    """Clean up old files"""
    try:
        # Clean up old plots
        plot_count = plotter.cleanup_old_plots(days_old=3)
        
        # Clean up old downloads
        download_count = downloader.cleanup_old_files(days_old=3)
        
        # Clean up old uploads
        upload_count = 0
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        cutoff_date = datetime.now() - timedelta(days=3)
        
        for upload_file in upload_dir.glob('*'):
            if datetime.fromtimestamp(upload_file.stat().st_mtime) < cutoff_date:
                upload_file.unlink()
                upload_count += 1
        
        flash(f'Cleanup completed: {plot_count} plots, {download_count} downloads, {upload_count} uploads removed', 'success')
        
    except Exception as e:
        flash(f'Cleanup error: {str(e)}', 'error')
    
    return redirect(url_for('index'))


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 500MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    flash('Page not found', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    flash('An internal error occurred', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("🚀 Starting CAMS Air Pollution Visualization App")
    print("📊 Available at: http://localhost:5050")
    print("🔧 CDS API Ready:", downloader.is_client_ready())
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5050)