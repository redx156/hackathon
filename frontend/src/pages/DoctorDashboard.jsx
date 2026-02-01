import { useState, useRef } from 'react';
import { AIChatbot } from '../components/AIChatbot';
import { useAppContext } from '../context/AppContext';

export function DoctorDashboard({ doctorName, doctorId, onReviewCase, onAnalyzeXRay, isAnalyzing, onLogout }) {
    const { getSubmissionsForDoctor, getDashboardCounts } = useAppContext();

    const [activeTab, setActiveTab] = useState('pending');
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const fileInputRef = useRef(null);

    // Get real submissions from context
    const doctorSubmissions = getSubmissionsForDoctor(doctorId);
    const counts = getDashboardCounts(doctorId);

    // Filter submissions based on tab
    const filteredCases = doctorSubmissions.filter(c => {
        if (activeTab === 'all') return true;
        return c.status === activeTab;
    });

    const getRiskBadgeColor = (riskLevel) => {
        switch (riskLevel) {
            case 'high': return 'bg-red-100 text-red-800';
            case 'medium': return 'bg-yellow-100 text-yellow-800';
            case 'low': return 'bg-green-100 text-green-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
            setSelectedFile(file);
        }
    };

    const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
            setSelectedFile(file);
        }
    };

    const handleAnalyzeXRay = () => {
        if (selectedFile && onAnalyzeXRay && !isAnalyzing) {
            onAnalyzeXRay(selectedFile);
            setSelectedFile(null);
        }
    };

    const openFilePicker = () => fileInputRef.current?.click();

    return (
        <div className="min-h-screen bg-transparent">
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Doctor Dashboard</h1>
                        <p className="text-gray-600">Welcome back, {doctorName}</p>
                    </div>
                    <button onClick={onLogout} className="px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">Logout</button>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                {/* Stats Cards - Use real counts from context */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-white rounded-xl shadow-sm p-6">
                        <div className="text-3xl font-bold text-blue-600">{counts.pending}</div>
                        <div className="text-gray-600">Pending Reviews</div>
                    </div>
                    <div className="bg-white rounded-xl shadow-sm p-6">
                        <div className="text-3xl font-bold text-green-600">{counts.reviewed}</div>
                        <div className="text-gray-600">Completed Reviews</div>
                    </div>
                    <div className="bg-white rounded-xl shadow-sm p-6">
                        <div className="text-3xl font-bold text-red-600">{counts.highRisk}</div>
                        <div className="text-gray-600">High Risk Cases</div>
                    </div>
                </div>

                {/* Cases List */}
                <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                    <div className="border-b border-gray-200 px-6">
                        <nav className="flex space-x-8">
                            {['pending', 'reviewed', 'all'].map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm capitalize transition-colors ${activeTab === tab ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
                                >
                                    {tab} {tab === 'pending' && counts.pending > 0 && <span className="ml-1 bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full text-xs">{counts.pending}</span>}
                                </button>
                            ))}
                        </nav>
                    </div>
                    <div className="divide-y divide-gray-200">
                        {filteredCases.length === 0 ? (
                            <div className="p-8 text-center text-gray-500">
                                {activeTab === 'pending'
                                    ? 'No pending reviews. Patient submissions will appear here.'
                                    : 'No cases found'}
                            </div>
                        ) : (
                            filteredCases.map(caseItem => (
                                <div key={caseItem.id} className="p-6 hover:bg-gray-50 transition-colors flex items-center justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center space-x-3">
                                            <h3 className="font-semibold text-gray-900">{caseItem.patientName}</h3>
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskBadgeColor(caseItem.riskLevel)}`}>
                                                {caseItem.riskLevel} risk
                                            </span>
                                            {caseItem.status === 'reviewed' && (
                                                <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                                    ✓ Reviewed
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-gray-500 mt-1">{caseItem.diagnosis} • Submitted {caseItem.submissionDate}</p>
                                    </div>
                                    <button
                                        onClick={() => onReviewCase(caseItem.id)}
                                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${caseItem.status === 'pending' ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
                                    >
                                        {caseItem.status === 'pending' ? 'Review' : 'View'}
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Upload Section */}
                <div className="bg-white rounded-xl shadow-sm p-8 mt-8">
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">Upload X-Ray for Analysis</h2>
                    <p className="text-gray-600 mb-6">Upload an X-ray image for independent AI-powered analysis</p>

                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={openFilePicker}
                        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragging ? 'border-blue-500 bg-blue-50' : selectedFile ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'}`}
                    >
                        <input ref={fileInputRef} type="file" accept="image/png,image/jpeg" onChange={handleFileSelect} className="hidden" />
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
                                <p className="text-sm text-gray-400 mt-4">Supported formats: PNG, JPG</p>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleAnalyzeXRay}
                        disabled={!selectedFile || isAnalyzing}
                        className={`w-full mt-6 py-3 px-6 rounded-lg font-semibold text-lg transition-colors ${selectedFile && !isAnalyzing ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}
                    >
                        {isAnalyzing ? 'Analyzing...' : 'Analyze X-Ray'}
                    </button>
                </div>
            </main>
            <AIChatbot />
        </div>
    );
}
