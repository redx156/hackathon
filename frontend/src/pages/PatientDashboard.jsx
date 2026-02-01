import { useState, useRef } from 'react';
import { AIChatbot } from '../components/AIChatbot';

export function PatientDashboard({ patientName, assignedDoctor, onUploadXRay, isAnalyzing, onLogout }) {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const fileInputRef = useRef(null);
    const [symptoms, setSymptoms] = useState({ fever: null, temperature: '', cough: '', shortnessOfBreath: null, chestPain: null, age: '', smokingStatus: null, bloodTestDone: null, wbcCount: '', crp: '', plateletCount: '' });
    const [symptomsSubmitted, setSymptomsSubmitted] = useState(false);

    const handleSymptomChange = (field, value) => { setSymptoms(prev => ({ ...prev, [field]: value })); setSymptomsSubmitted(false); };
    const handleFileSelect = (e) => { const file = e.target.files[0]; if (file) setSelectedFile(file); };
    const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => { e.preventDefault(); setIsDragging(false); const file = e.dataTransfer.files[0]; if (file) setSelectedFile(file); };

    // REAL API CALL - triggers upload
    const handleUpload = () => {
        if (selectedFile && !isAnalyzing) {
            onUploadXRay(selectedFile, symptoms);
            setSelectedFile(null); // Clear after starting
        }
    };

    const openFilePicker = () => fileInputRef.current?.click();

    return (
        <div className="min-h-screen bg-transparent">
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Patient Dashboard</h1>
                        <p className="text-gray-600">Welcome, {patientName}</p>
                    </div>
                    <button onClick={onLogout} className="px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">Logout</button>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-4 py-8">
                <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
                    <h2 className="text-lg font-semibold text-gray-900 mb-4">Your Assigned Doctor</h2>
                    <div className="flex items-center space-x-4">
                        <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                            <span className="text-2xl">👨⚕️</span>
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900">{assignedDoctor.name}</h3>
                            <p className="text-gray-600">{assignedDoctor.specialization}</p>
                            <p className="text-sm text-gray-500">{assignedDoctor.hospital}</p>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-xl shadow-sm p-8">
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">Upload X-Ray for Analysis</h2>
                    <p className="text-gray-600 mb-6">Upload your chest X-ray image to receive AI-powered analysis</p>

                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={openFilePicker}
                        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragging ? 'border-blue-500 bg-blue-50' : selectedFile ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'}`}
                    >
                        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
                        {selectedFile ? (
                            <div>
                                <div className="text-4xl mb-4">✅</div>
                                <p className="text-lg font-medium text-gray-900">{selectedFile.name}</p>
                                <p className="text-sm text-gray-500 mt-1">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                                <p className="text-sm text-green-600 mt-2">Click to select a different file</p>
                            </div>
                        ) : (
                            <div>
                                <div className="text-4xl mb-4">📤</div>
                                <p className="text-lg font-medium text-gray-900">Drag and drop your X-ray image here</p>
                                <p className="text-gray-500 mt-1">or click to browse files</p>
                                <p className="text-sm text-gray-400 mt-4">Supported formats: JPG, PNG, DICOM</p>
                            </div>
                        )}
                    </div>

                    {selectedFile && (
                        <button
                            onClick={handleUpload}
                            disabled={isAnalyzing}
                            className={`w-full mt-6 py-3 px-6 rounded-lg font-semibold text-lg transition-colors shadow-md ${isAnalyzing ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg'}`}
                        >
                            {isAnalyzing ? 'Analyzing...' : 'Upload and Analyze'}
                        </button>
                    )}
                </div>

                <div className="bg-white rounded-xl shadow-sm p-8 mt-8">
                    <div className="flex items-start justify-between mb-2">
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">Additional Patient Symptoms (Optional)</h2>
                            <p className="text-gray-600 mt-2">Provide additional clinical details to support pneumonia analysis</p>
                        </div>
                        <button
                            onClick={() => setSymptomsSubmitted(true)}
                            disabled={symptomsSubmitted}
                            className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 flex-shrink-0 ${symptomsSubmitted ? 'bg-green-100 text-green-700 cursor-default' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
                        >
                            {symptomsSubmitted ? '✓ Symptoms Saved' : 'Submit Symptoms'}
                        </button>
                    </div>

                    <div className="space-y-6">
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Fever</label><div className="flex space-x-4"><label className="flex items-center"><input type="radio" name="fever" value="yes" checked={symptoms.fever === 'yes'} onChange={(e) => handleSymptomChange('fever', e.target.value)} className="mr-2" />Yes</label><label className="flex items-center"><input type="radio" name="fever" value="no" checked={symptoms.fever === 'no'} onChange={(e) => handleSymptomChange('fever', e.target.value)} className="mr-2" />No</label></div></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Temperature (°C)</label><input type="number" step="0.1" placeholder="e.g., 38.5" value={symptoms.temperature} onChange={(e) => handleSymptomChange('temperature', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" /></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Cough</label><select value={symptoms.cough} onChange={(e) => handleSymptomChange('cough', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"><option value="">Select...</option><option value="no">No</option><option value="mild">Mild</option><option value="severe">Severe</option></select></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Shortness of Breath</label><div className="flex space-x-4"><label className="flex items-center"><input type="radio" name="shortnessOfBreath" value="yes" checked={symptoms.shortnessOfBreath === 'yes'} onChange={(e) => handleSymptomChange('shortnessOfBreath', e.target.value)} className="mr-2" />Yes</label><label className="flex items-center"><input type="radio" name="shortnessOfBreath" value="no" checked={symptoms.shortnessOfBreath === 'no'} onChange={(e) => handleSymptomChange('shortnessOfBreath', e.target.value)} className="mr-2" />No</label></div></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Chest Pain</label><div className="flex space-x-4"><label className="flex items-center"><input type="radio" name="chestPain" value="yes" checked={symptoms.chestPain === 'yes'} onChange={(e) => handleSymptomChange('chestPain', e.target.value)} className="mr-2" />Yes</label><label className="flex items-center"><input type="radio" name="chestPain" value="no" checked={symptoms.chestPain === 'no'} onChange={(e) => handleSymptomChange('chestPain', e.target.value)} className="mr-2" />No</label></div></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Age</label><input type="number" min="0" max="120" placeholder="Enter age" value={symptoms.age} onChange={(e) => handleSymptomChange('age', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" /></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Do you smoke?</label><div className="flex space-x-4"><label className="flex items-center"><input type="radio" name="smokingStatus" value="yes" checked={symptoms.smokingStatus === 'yes'} onChange={(e) => handleSymptomChange('smokingStatus', e.target.value)} className="mr-2" />Yes</label><label className="flex items-center"><input type="radio" name="smokingStatus" value="no" checked={symptoms.smokingStatus === 'no'} onChange={(e) => handleSymptomChange('smokingStatus', e.target.value)} className="mr-2" />No</label></div></div>
                        <div><label className="block text-sm font-medium text-gray-700 mb-2">Blood test available?</label><div className="flex space-x-4"><label className="flex items-center"><input type="radio" name="bloodTestDone" value="yes" checked={symptoms.bloodTestDone === 'yes'} onChange={(e) => handleSymptomChange('bloodTestDone', e.target.value)} className="mr-2" />Yes</label><label className="flex items-center"><input type="radio" name="bloodTestDone" value="no" checked={symptoms.bloodTestDone === 'no'} onChange={(e) => handleSymptomChange('bloodTestDone', e.target.value)} className="mr-2" />No</label></div></div>
                        {symptoms.bloodTestDone === 'yes' && (<div className="pl-4 border-l-2 border-blue-200 space-y-6"><div><label className="block text-sm font-medium text-gray-700 mb-2">Elevated WBC Count (cells/µL)</label><input type="number" placeholder="11,000 – 20,000" value={symptoms.wbcCount} onChange={(e) => handleSymptomChange('wbcCount', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" /></div><div><label className="block text-sm font-medium text-gray-700 mb-2">Elevated CRP (mg/L)</label><input type="number" placeholder="10 – 200" value={symptoms.crp} onChange={(e) => handleSymptomChange('crp', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" /></div><div><label className="block text-sm font-medium text-gray-700 mb-2">Platelet Count (cells/µL)</label><input type="number" placeholder="150,000 – 450,000" value={symptoms.plateletCount} onChange={(e) => handleSymptomChange('plateletCount', e.target.value)} className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" /></div></div>)}
                    </div>
                </div>

                <div className="mt-8 bg-blue-50 rounded-xl p-6">
                    <h3 className="font-semibold text-blue-900 mb-2">How it works</h3>
                    <ol className="text-blue-800 space-y-2">
                        <li>1. Upload your chest X-ray image</li>
                        <li>2. Our AI analyzes the image for abnormalities</li>
                        <li>3. Your doctor reviews the AI findings</li>
                        <li>4. You receive the final diagnosis from your doctor</li>
                    </ol>
                </div>
            </main>
            <AIChatbot />
        </div>
    );
}
