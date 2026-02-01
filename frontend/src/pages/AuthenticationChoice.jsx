export function AuthenticationChoice({ role, onRegister, onLogin, onBack }) {
    const roleLabel = role === 'doctor' ? 'Doctor' : 'Patient';

    return (
        <div className="min-h-screen bg-transparent flex items-center justify-center p-4">
            <div className="max-w-md w-full">
                <button onClick={onBack} className="flex items-center text-gray-600 hover:text-gray-900 mb-8 transition-colors"><span className="mr-2">←</span>Back to role selection</button>
                <div className="text-center mb-10">
                    <div className="text-5xl mb-4">{role === 'doctor' ? '👨‍⚕️' : '🩺'}</div>
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">{roleLabel} Portal</h1>
                    <p className="text-gray-600">Sign in to your account or create a new one</p>
                </div>
                <div className="space-y-4">
                    <button onClick={onLogin} className="w-full py-4 px-6 bg-blue-600 text-white rounded-xl font-semibold text-lg hover:bg-blue-700 transition-colors shadow-md hover:shadow-lg">Login</button>
                    <button onClick={onRegister} className="w-full py-4 px-6 bg-white text-blue-600 border-2 border-blue-600 rounded-xl font-semibold text-lg hover:bg-blue-50 transition-colors">Create Account</button>
                </div>
                <p className="text-center text-gray-500 mt-8 text-sm">By continuing, you agree to our Terms of Service and Privacy Policy</p>
            </div>
        </div>
    );
}
