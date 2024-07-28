import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { v4 as uuidv4 } from 'uuid';

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loadingMessages, setLoadingMessages] = useState(false);

  const chatEnvRef = useRef(null);

  const convertToIST = (gmtDateString) => {
    const gmtDate = new Date(gmtDateString);
    const istDate = new Date(gmtDate.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const day = istDate.toLocaleDateString('en-GB', { weekday: 'short' });
    const date = istDate.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    const time = istDate.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    return `${day}, ${date} ${time} IST`;
  };

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const sanitizedHumanMessage = DOMPurify.sanitize(marked(input));
    const newMessage = {
      content: sanitizedHumanMessage,
      sentBy: 'user',
      time: convertToIST(new Date().toUTCString()),
      id: uuidv4(),
    };

    setMessages([...messages, newMessage]);
    setInput('');

    setLoadingMessages(true);

    try {
      const response = await axios.post(
        'http://127.0.0.1:5000/chat',
        {
          job_description: input,
        }
      );

      const responses = response.data.responses;
      const consolidatedResponses = 'Matching candidate profiles:\n\n' + responses.join('\n\n\n');
      const aiMessage = {
        content: DOMPurify.sanitize(marked(consolidatedResponses)),
        sentBy: 'ai',
        time: convertToIST(new Date().toUTCString()),
        id: uuidv4(),
      };

      setMessages([...messages, newMessage, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages([
        ...messages,
        { content: 'Sorry, there was an error processing your request.', sentBy: 'bot', time: convertToIST(new Date().toUTCString()), id: uuidv4() },
      ]);
    } finally {
      setLoadingMessages(false);
    }
  };

  return (
    <div className="fixed inset-0 flex flex-col font-sans bg-[#17171a] text-[#d6d5de]">
      <div className="flex justify-between items-center text-white p-4">
        <h3 className="font-semibold font-sans text-xl text-[#d6d5de]">Job Description RAG</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4" ref={chatEnvRef}>
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-3xl font-semibold">Welcome to the Chatbot</p>
          </div>
        )}
        <ul className="flex flex-col w-full space-y-2">
          {messages.map((message) => (
            <motion.li
              key={message.id}
              initial={{ opacity: 0.8, x: message.sentBy === 'user' ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className={`p-3 rounded-lg ${message.sentBy === 'user' ? 'bg-[#d6d5de] font-sans leading-6 text-black font-semibold self-end' : 'bg-[#2f2e35] text-[#d6d5de] self-start'}`}
            >
              <div dangerouslySetInnerHTML={{ __html: message.content }} />
              <span className="block text-xs mt-1 text-right">{message.time}</span>
            </motion.li>
          ))}
          {loadingMessages && (
            <li className="flex justify-start p-5">
              <span>Loading...</span>
            </li>
          )}
        </ul>
      </div>
      <div className="p-4 border-t h-32 border-gray-700 flex">
        <textarea
          className="flex-grow p-2 rounded bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-white"
          placeholder="Enter Job Description here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <button
          onClick={sendMessage}
          className="ml-2 p-2 bg-[#fafafa] rounded h-1/2 items-center w-12 my-auto hover:bg-[#e3e3e4] transition duration-300"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16px" height="16px" viewBox="0 0 24 24" className="text-black mx-auto">
            <path fill="currentColor" d="m3.4 20.4l17.45-7.48a1 1 0 0 0 0-1.84L3.4 3.6a.993.993 0 0 0-1.39.91L2 9.12c0 .5.37.93.87.99L17 12L2.87 13.88c-.5.07-.87.5-.87 1l.01 4.61c0 .71.73 1.2 1.39.91"></path>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ChatBot;
