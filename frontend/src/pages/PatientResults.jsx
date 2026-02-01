import { AIChatbot } from '../components/AIChatbot';

export function PatientResults({ patientName, xrayImageUrl, analysisResult, error, assignedDoctorName, onBack }) {
    // Show error state if API call failed
    if (error) {
        return (
            <div className="min-h-screen bg-transparent">
                <header className="bg-white shadow-sm">
                    <div className="max-w-7xl mx-auto px-4 py-4">
                        <button onClick={onBack} className="flex items-center text-gray-600 hover:text-gray-900 transition-colors">
                            <span className="mr-2">←</span>Back to Dashboard
                        </button>
                    </div>
                </header>
                <main className="max-w-3xl mx-auto px-4 py-8">
                    <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
                        <div className="text-5xl mb-4">⚠️</div>
                        <h1 className="text-2xl font-bold text-red-900 mb-2">Upload Failed</h1>
                        <p className="text-red-700 mb-6">{error}</p>
                        <button
                            onClick={onBack}
                            className="px-6 py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-colors"
                        >
                            Try Again
                        </button>
                    </div>
                </main>
                <AIChatbot />
            </div>
        );
    }

    // Show success state with real results
    return (
        <div className="min-h-screen bg-transparent">
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <button onClick={onBack} className="flex items-center text-gray-600 hover:text-gray-900 transition-colors">
                        <span className="mr-2">←</span>Back to Dashboard
                    </button>
                </div>
            </header>

            <main className="max-w-3xl mx-auto px-4 py-8">
                <div className="bg-green-50 border border-green-200 rounded-xl p-6 mb-8 text-center">
                    <div className="text-4xl mb-4">✅</div>
                    <h1 className="text-2xl font-bold text-green-900 mb-2">X-Ray Uploaded Successfully!</h1>
                    <p className="text-green-700">Your X-ray has been analyzed by our AI</p>
                </div>

                {/* Image Quality Warning */}
                {analysisResult?.lowImageQuality && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6">
                        <div className="flex items-center">
                            <span className="text-2xl mr-3">⚠️</span>
                            <div>
                                <h3 className="font-semibold text-yellow-900">Image Quality Notice</h3>
                                <p className="text-yellow-700 text-sm">The uploaded image may have quality issues. Your doctor will review this carefully.</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* AI Analysis Summary - for patient view */}
                {analysisResult && (
                    <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                        <h2 className="font-semibold text-gray-900 mb-4">AI Analysis Summary</h2>
                        <div className="flex items-center space-x-4 mb-4">
                            <div className={`text-xl font-bold ${analysisResult.prediction === 'Pneumonia' ? 'text-red-600' : 'text-green-600'}`}>
                                {analysisResult.prediction}
                            </div>
                            <div className="flex items-center">
                                <span className="text-sm text-gray-500 mr-2">Confidence:</span>
                                <span className="font-semibold text-blue-600">{analysisResult.confidencePercent}%</span>
                            </div>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-sm font-medium inline-block ${analysisResult.riskLevel?.toLowerCase().includes('high') ? 'bg-red-100 text-red-800' :
                                analysisResult.riskLevel?.toLowerCase().includes('medium') ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                            }`}>
                            {analysisResult.riskLevel}
                        </div>
                        {analysisResult.note && (
                            <p className="text-gray-700 mt-4 text-sm">{analysisResult.note}</p>
                        )}
                    </div>
                )}

                {/* X-Ray Images */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                        <div className="px-6 py-4 border-b border-gray-200">
                            <h2 className="font-semibold text-gray-900">Your Uploaded X-Ray</h2>
                        </div>
                        <div className="p-4">
                            <img
                                src={xrayImageUrl}
                                alt="Uploaded X-Ray"
                                className="w-full h-64 object-contain rounded-lg bg-gray-100"
                            />
                        </div>
                    </div>

                    {analysisResult?.heatmapDataUri && (
                        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-200">
                                <h2 className="font-semibold text-gray-900">AI Analysis Heatmap</h2>
                            </div>
                            <div className="p-4">
                                <img
                                    src={analysisResult.heatmapDataUri}
                                    alt="AI Heatmap Analysis"
                                    className="w-full h-64 object-contain rounded-lg bg-gray-100"
                                />
                            </div>
                        </div>
                    )}
                </div>

                <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                    <h2 className="font-semibold text-gray-900 mb-4">Review Status</h2>
                    <div className="space-y-4">
                        <div className="flex items-center">
                            <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mr-4">
                                <span className="text-green-600">✓</span>
                            </div>
                            <div>
                                <div className="font-medium text-gray-900">AI Analysis Complete</div>
                                <div className="text-sm text-gray-500">Initial screening done</div>
                            </div>
                        </div>
                        <div className="flex items-center">
                            <div className="w-10 h-10 bg-yellow-100 rounded-full flex items-center justify-center mr-4">
                                <span className="text-yellow-600">⏳</span>
                            </div>
                            <div>
                                <div className="font-medium text-gray-900">Pending Doctor Review</div>
                                <div className="text-sm text-gray-500">{assignedDoctorName} will review your results</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-blue-50 rounded-xl p-6">
                    <h3 className="font-semibold text-blue-900 mb-2">What happens next?</h3>
                    <ul className="text-blue-800 space-y-2">
                        <li>• Your doctor will review the AI analysis results</li>
                        <li>• They will provide their professional diagnosis</li>
                        <li>• You will be notified when the review is complete</li>
                        <li>• You can view the full results once approved</li>
                    </ul>
                </div>

                <div className="mt-6 text-center text-sm text-gray-500">Patient: {patientName}</div>
            </main>
            <AIChatbot />
        </div>
    );
}
