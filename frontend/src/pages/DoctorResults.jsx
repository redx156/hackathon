import { useState } from 'react';
import { AIChatbot } from '../components/AIChatbot';

export function DoctorResults({ caseData, doctorName, error, onMarkReviewed, onBack }) {
    const [notes, setNotes] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const getRiskColor = (riskLevel) => {
        switch (riskLevel) {
            case 'high': return 'text-red-600 bg-red-100';
            case 'medium': return 'text-yellow-600 bg-yellow-100';
            case 'low': return 'text-green-600 bg-green-100';
            default: return 'text-gray-600 bg-gray-100';
        }
    };

    const handleSubmit = async () => {
        setIsSubmitting(true);
        await new Promise(resolve => setTimeout(resolve, 800));
        onMarkReviewed(caseData.id, notes);
    };

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
                        <h1 className="text-2xl font-bold text-red-900 mb-2">Analysis Failed</h1>
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

    return (
        <div className="min-h-screen bg-transparent">
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <button onClick={onBack} className="flex items-center text-gray-600 hover:text-gray-900 transition-colors">
                        <span className="mr-2">←</span>Back to Dashboard
                    </button>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-8">
                <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">{caseData.patientName}</h1>
                            <p className="text-gray-600">Submitted on {caseData.submissionDate}</p>
                        </div>
                        <span className={`px-4 py-2 rounded-full font-medium ${getRiskColor(caseData.riskLevel)}`}>
                            {caseData.riskLevel.charAt(0).toUpperCase() + caseData.riskLevel.slice(1)} Risk
                        </span>
                    </div>
                </div>

                {/* Image Quality Warning */}
                {caseData.lowImageQuality && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6">
                        <div className="flex items-center">
                            <span className="text-2xl mr-3">⚠️</span>
                            <div>
                                <h3 className="font-semibold text-yellow-900">Image Quality Notice</h3>
                                <p className="text-yellow-700 text-sm">The uploaded image may have quality issues. Results should be interpreted with caution.</p>
                            </div>
                        </div>
                    </div>
                )}

                {caseData.patientSymptoms && (
                    <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                        <h2 className="font-semibold text-gray-900 mb-4">Patient-Reported Symptoms</h2>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            {caseData.patientSymptoms.fever && <div><span className="text-gray-500">Fever:</span><span className="ml-2 font-medium capitalize">{caseData.patientSymptoms.fever}</span></div>}
                            {caseData.patientSymptoms.temperature && <div><span className="text-gray-500">Temperature:</span><span className="ml-2 font-medium">{caseData.patientSymptoms.temperature} °C</span></div>}
                            {caseData.patientSymptoms.cough && <div><span className="text-gray-500">Cough:</span><span className="ml-2 font-medium capitalize">{caseData.patientSymptoms.cough}</span></div>}
                            {caseData.patientSymptoms.shortnessOfBreath && <div><span className="text-gray-500">Shortness of Breath:</span><span className="ml-2 font-medium capitalize">{caseData.patientSymptoms.shortnessOfBreath}</span></div>}
                            {caseData.patientSymptoms.chestPain && <div><span className="text-gray-500">Chest Pain:</span><span className="ml-2 font-medium capitalize">{caseData.patientSymptoms.chestPain}</span></div>}
                            {caseData.patientSymptoms.age && <div><span className="text-gray-500">Age:</span><span className="ml-2 font-medium">{caseData.patientSymptoms.age}</span></div>}
                            {caseData.patientSymptoms.smokingStatus && <div><span className="text-gray-500">Smoking:</span><span className="ml-2 font-medium capitalize">{caseData.patientSymptoms.smokingStatus}</span></div>}
                            {caseData.patientSymptoms.bloodTestDone === 'yes' && caseData.patientSymptoms.wbcCount && <div><span className="text-gray-500">WBC:</span><span className="ml-2 font-medium">{caseData.patientSymptoms.wbcCount}</span></div>}
                            {caseData.patientSymptoms.bloodTestDone === 'yes' && caseData.patientSymptoms.crp && <div><span className="text-gray-500">CRP:</span><span className="ml-2 font-medium">{caseData.patientSymptoms.crp}</span></div>}
                            {caseData.patientSymptoms.bloodTestDone === 'yes' && caseData.patientSymptoms.plateletCount && <div><span className="text-gray-500">Platelets:</span><span className="ml-2 font-medium">{caseData.patientSymptoms.plateletCount}</span></div>}
                        </div>
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                        <div className="px-6 py-4 border-b border-gray-200">
                            <h2 className="font-semibold text-gray-900">Original X-Ray</h2>
                        </div>
                        <div className="p-4">
                            <img
                                src={caseData.xrayImageUrl}
                                alt="Original X-Ray"
                                className="w-full h-64 object-contain rounded-lg bg-gray-100"
                            />
                        </div>
                    </div>
                    <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                        <div className="px-6 py-4 border-b border-gray-200">
                            <h2 className="font-semibold text-gray-900">AI Heatmap Analysis</h2>
                        </div>
                        <div className="p-4">
                            <img
                                src={caseData.heatmapImageUrl}
                                alt="AI Heatmap"
                                className="w-full h-64 object-contain rounded-lg bg-gray-100"
                            />
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                    <h2 className="font-semibold text-gray-900 mb-4">AI Diagnosis Summary</h2>
                    <div className="flex items-center space-x-4 mb-4">
                        <div className="text-2xl font-bold text-gray-900">{caseData.diagnosis}</div>
                        <div className="flex items-center">
                            <span className="text-sm text-gray-500 mr-2">Confidence:</span>
                            <span className="font-semibold text-blue-600">{caseData.confidence}%</span>
                        </div>
                    </div>
                    <p className="text-gray-700">{caseData.aiObservation}</p>
                </div>

                <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                    <h2 className="font-semibold text-gray-900 mb-4">Doctor's Notes</h2>
                    <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        rows={4}
                        placeholder="Add your clinical notes and observations here..."
                        className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all resize-none"
                    />
                </div>

                <div className="flex justify-end space-x-4">
                    <button onClick={onBack} className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50">
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting}
                        className={`px-6 py-3 rounded-lg font-medium ${isSubmitting ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 text-white hover:bg-green-700'}`}
                    >
                        {isSubmitting ? 'Submitting...' : 'Mark as Reviewed'}
                    </button>
                </div>

                <p className="text-center text-sm text-gray-500 mt-4">Reviewed by: {doctorName}</p>
            </main>
            <AIChatbot />
        </div>
    );
}
