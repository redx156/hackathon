import { useState } from 'react';

export function AIChatbot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([{ id: 1, sender: 'bot', text: 'Hello! I\'m your AI assistant. How can I help you today?' }]);
    const [inputValue, setInputValue] = useState('');

    const toggleChat = () => setIsOpen(!isOpen);

    const handleSendMessage = () => {
        if (!inputValue.trim()) return;
        const userMessage = { id: messages.length + 1, sender: 'user', text: inputValue };
        const botResponse = { id: messages.length + 2, sender: 'bot', text: 'Thank you for your message. This is a demo response. In the full version, I will be connected to an AI service to provide helpful guidance about X-ray analysis and the platform.' };
        setMessages([...messages, userMessage, botResponse]);
        setInputValue('');
    };

    const handleKeyPress = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } };

    return (
        <>
            {!isOpen && (
                <button onClick={toggleChat} className="fixed bottom-6 right-6 w-14 h-14 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg hover:shadow-xl transition-all flex items-center justify-center z-50" aria-label="Open AI Chatbot">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
                </button>
            )}
            {isOpen && (
                <div className="fixed bottom-6 right-6 w-96 h-[500px] bg-white rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden">
                    <div className="bg-blue-600 text-white px-6 py-4 flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center"><span className="text-lg">🤖</span></div>
                            <div><h3 className="font-semibold">AI Assistant</h3><p className="text-xs text-blue-200">Online • Here to help</p></div>
                        </div>
                        <button onClick={toggleChat} className="text-white hover:bg-blue-500 rounded-full p-1 transition-colors" aria-label="Close chatbot">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
                        {messages.map((message) => (
                            <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[80%] px-4 py-3 rounded-2xl ${message.sender === 'user' ? 'bg-blue-600 text-white rounded-br-sm' : 'bg-white text-gray-800 shadow-sm rounded-bl-sm'}`}>
                                    <p className="text-sm">{message.text}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="border-t border-gray-200 p-4 bg-white">
                        <div className="flex items-center space-x-2">
                            <input type="text" value={inputValue} onChange={(e) => setInputValue(e.target.value)} onKeyPress={handleKeyPress} placeholder="Type a message..." className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all text-sm" />
                            <button onClick={handleSendMessage} disabled={!inputValue.trim()} className={`p-2 rounded-full transition-colors ${inputValue.trim() ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`} aria-label="Send message">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                            </button>
                        </div>
                        <p className="text-xs text-gray-400 mt-2 text-center">AI chatbot is for guidance only</p>
                    </div>
                </div>
            )}
        </>
    );
}
