import { useState } from 'react';

export function RoleSelection({ onContinue }) {
    const [selectedRole, setSelectedRole] = useState(null);

    const handleRoleClick = (role) => setSelectedRole(role);
    const handleContinue = () => { if (selectedRole) onContinue(selectedRole); };

    return (
        <div className="min-h-screen bg-transparent flex items-center justify-center p-4">
            <div className="max-w-4xl w-full">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-gray-900 mb-4">Welcome to X-Ray Diagnosis</h1>
                    <p className="text-lg text-gray-600">AI-powered chest X-ray analysis for faster, more accurate diagnoses</p>
                </div>
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                    <button onClick={() => handleRoleClick('doctor')} className={`p-8 rounded-2xl border-2 transition-all duration-200 text-left ${selectedRole === 'doctor' ? 'border-blue-500 bg-blue-50 shadow-lg' : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-md'}`}>
                        <div className="text-5xl mb-4">👨‍⚕️</div>
                        <h2 className="text-2xl font-semibold text-gray-900 mb-2">I'm a Doctor</h2>
                        <p className="text-gray-600">Review AI-analyzed X-rays, provide diagnoses, and manage patient cases</p>
                    </button>
                    <button onClick={() => handleRoleClick('patient')} className={`p-8 rounded-2xl border-2 transition-all duration-200 text-left ${selectedRole === 'patient' ? 'border-blue-500 bg-blue-50 shadow-lg' : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-md'}`}>
                        <div className="text-5xl mb-4">🩺</div>
                        <h2 className="text-2xl font-semibold text-gray-900 mb-2">I'm a Patient</h2>
                        <p className="text-gray-600">Upload X-rays for AI analysis and receive results from your doctor</p>
                    </button>
                </div>
                <div className="text-center">
                    <button onClick={handleContinue} disabled={!selectedRole} className={`px-8 py-3 rounded-lg font-semibold text-lg transition-all duration-200 ${selectedRole ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}>Continue</button>
                </div>
            </div>
        </div>
    );
}
