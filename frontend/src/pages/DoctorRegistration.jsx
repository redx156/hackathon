import { useState } from 'react';
import { useAppContext } from '../context/AppContext';

export function DoctorRegistration({ onSuccess, onBack }) {
    // Get addDoctor from context to immediately add doctor to shared list
    const { addDoctor } = useAppContext();

    const [formData, setFormData] = useState({ fullName: '', email: '', hospital: '', specialization: '', password: '', confirmPassword: '' });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (e) => { const { name, value } = e.target; setFormData(prev => ({ ...prev, [name]: value })); setError(''); };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (formData.password !== formData.confirmPassword) { setError('Passwords do not match'); return; }
        if (formData.password.length < 6) { setError('Password must be at least 6 characters'); return; }
        setIsLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Add doctor to context immediately so they appear in patient dropdown
        const newDoctor = addDoctor(formData);
        console.log('New doctor registered:', newDoctor);

        setIsLoading(false);
        // Pass form data + new doctor ID to parent
        onSuccess({ ...formData, doctorId: newDoctor.id });
    };

    return (
        <div className="min-h-screen bg-transparent flex items-center justify-center p-4">
            <div className="max-w-lg w-full bg-white rounded-2xl shadow-xl p-8">
                <button onClick={onBack} className="flex items-center text-gray-600 hover:text-gray-900 mb-6 transition-colors"><span className="mr-2">←</span>Back</button>
                <div className="text-center mb-8"><div className="text-4xl mb-3">👨‍⚕️</div><h1 className="text-2xl font-bold text-gray-900">Doctor Registration</h1><p className="text-gray-600 mt-2">Create your professional account</p></div>
                <form onSubmit={handleSubmit} className="space-y-5">
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label><input type="text" name="fullName" value={formData.fullName} onChange={handleChange} required placeholder="Dr. John Smith" className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" /></div>
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label><input type="email" name="email" value={formData.email} onChange={handleChange} required placeholder="doctor@hospital.com" className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" /></div>
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Hospital / Clinic</label><input type="text" name="hospital" value={formData.hospital} onChange={handleChange} required placeholder="City General Hospital" className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" /></div>
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Specialization</label><select name="specialization" value={formData.specialization} onChange={handleChange} required className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"><option value="">Select specialization</option><option value="Radiology">Radiology</option><option value="Pulmonology">Pulmonology</option><option value="Internal Medicine">Internal Medicine</option><option value="General Practice">General Practice</option></select></div>
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Password</label><input type="password" name="password" value={formData.password} onChange={handleChange} required placeholder="••••••••" className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" /></div>
                    <div><label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label><input type="password" name="confirmPassword" value={formData.confirmPassword} onChange={handleChange} required placeholder="••••••••" className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" /></div>
                    {error && <div className="text-red-600 text-sm bg-red-50 p-3 rounded-lg">{error}</div>}
                    <button type="submit" disabled={isLoading} className={`w-full py-3 px-6 rounded-lg font-semibold text-lg transition-all ${isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'}`}>{isLoading ? 'Creating Account...' : 'Create Account'}</button>
                </form>
            </div>
        </div>
    );
}

