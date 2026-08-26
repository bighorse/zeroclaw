import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Send,
  Bot,
  User,
  AlertCircle,
  Copy,
  Check,
  Paperclip,
  FileText,
  Loader2,
  X,
} from 'lucide-react';
import type { WsMessage } from '@/types/api';
import { WebSocketClient } from '@/lib/ws';
import { uploadFile } from '@/lib/api';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: Date;
}

type AttachmentStatus = 'uploading' | 'ready' | 'error';

interface Attachment {
  id: string;
  /** Display name; replaced by the stored name once the upload succeeds. */
  name: string;
  size: number;
  isImage: boolean;
  status: AttachmentStatus;
  /** Object URL used for the local thumbnail; null for non-images. */
  previewUrl: string | null;
  /** Absolute path on the agent host, set once the upload succeeds. */
  path: string | null;
  error: string | null;
}

/**
 * Prefix used for non-image attachments.
 *
 * `/ws/chat` runs a single provider turn with no tools (see
 * `src/gateway/ws.rs`), so the agent cannot open the file — only the path
 * travels with the message. The wording says so explicitly rather than
 * pretending the file was read.
 */
const NON_IMAGE_NOTE = 'Attached file (path only, this chat has no file tools):';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Build the outgoing message body.
 *
 * Images become `[IMAGE:<absolute path>]` markers on their own leading lines,
 * separated from the typed text by a blank line — the same shape the Telegram
 * channel produces for photo captions. The gateway already runs
 * `multimodal::prepare_messages_for_provider` on the WebSocket path, so the
 * markers are inlined for the provider with no backend change.
 */
function composeOutgoing(text: string, attachments: Attachment[]): string {
  const lines: string[] = [];
  for (const attachment of attachments) {
    if (attachment.path === null) continue;
    lines.push(
      attachment.isImage
        ? `[IMAGE:${attachment.path}]`
        : `${NON_IMAGE_NOTE} ${attachment.path}`,
    );
  }

  const header = lines.join('\n');
  const trimmed = text.trim();
  if (!header) return trimmed;
  return trimmed ? `${header}\n\n${trimmed}` : header;
}

export default function AgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const wsRef = useRef<WebSocketClient | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const attachmentsRef = useRef<Attachment[]>([]);
  const dragDepthRef = useRef(0);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const pendingContentRef = useRef('');

  const uploading = attachments.some((a) => a.status === 'uploading');
  const readyAttachments = attachments.filter((a) => a.status === 'ready');
  const hasNonImageAttachment = attachments.some((a) => !a.isImage);
  const canSend =
    !uploading && (input.trim().length > 0 || readyAttachments.length > 0);

  useEffect(() => {
    const ws = new WebSocketClient();

    ws.onOpen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onClose = () => {
      setConnected(false);
    };

    ws.onError = () => {
      setError('Connection error. Attempting to reconnect...');
    };

    ws.onMessage = (msg: WsMessage) => {
      switch (msg.type) {
        case 'chunk':
          setTyping(true);
          pendingContentRef.current += msg.content ?? '';
          break;

        case 'message':
        case 'done': {
          const content = msg.full_response ?? msg.content ?? pendingContentRef.current;
          if (content) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: 'agent',
                content,
                timestamp: new Date(),
              },
            ]);
          }
          pendingContentRef.current = '';
          setTyping(false);
          break;
        }

        case 'tool_call':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Tool Call] ${msg.name ?? 'unknown'}(${JSON.stringify(msg.args ?? {})})`,
              timestamp: new Date(),
            },
          ]);
          break;

        case 'tool_result':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Tool Result] ${msg.output ?? ''}`,
              timestamp: new Date(),
            },
          ]);
          break;

        case 'error':
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'agent',
              content: `[Error] ${msg.message ?? 'Unknown error'}`,
              timestamp: new Date(),
            },
          ]);
          setTyping(false);
          pendingContentRef.current = '';
          break;
      }
    };

    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  // Mirror attachments into a ref so the unmount cleanup below sees the latest
  // list without re-registering the effect on every change.
  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  useEffect(
    () => () => {
      for (const attachment of attachmentsRef.current) {
        if (attachment.previewUrl) {
          URL.revokeObjectURL(attachment.previewUrl);
        }
      }
    },
    [],
  );

  const handleFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    const pending = files.map((file) => {
      const isImage = file.type.startsWith('image/');
      const attachment: Attachment = {
        id: crypto.randomUUID(),
        name: file.name || (isImage ? 'pasted-image' : 'file'),
        size: file.size,
        isImage,
        status: 'uploading',
        previewUrl: isImage ? URL.createObjectURL(file) : null,
        path: null,
        error: null,
      };
      return { file, attachment };
    });

    setAttachments((prev) => [...prev, ...pending.map((p) => p.attachment)]);

    await Promise.all(
      pending.map(async ({ file, attachment }) => {
        try {
          const uploaded = await uploadFile(file);
          setAttachments((prev) =>
            prev.map(
              (a): Attachment =>
                a.id === attachment.id
                  ? {
                      ...a,
                      status: 'ready',
                      path: uploaded.path,
                      name: uploaded.filename ?? a.name,
                      error: null,
                    }
                  : a,
            ),
          );
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Upload failed';
          setAttachments((prev) =>
            prev.map(
              (a): Attachment =>
                a.id === attachment.id
                  ? { ...a, status: 'error', error: message }
                  : a,
            ),
          );
        }
      }),
    );
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (list && list.length > 0) {
      void handleFiles(Array.from(list));
    }
    // Reset so picking the same file twice still fires onChange.
    e.target.value = '';
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(e.clipboardData.files);
    if (files.length === 0) return;
    e.preventDefault();
    void handleFiles(files);
  };

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const next: Attachment[] = [];
      for (const attachment of prev) {
        if (attachment.id === id) {
          if (attachment.previewUrl) {
            URL.revokeObjectURL(attachment.previewUrl);
          }
        } else {
          next.push(attachment);
        }
      }
      return next;
    });
  }, []);

  const isFileDrag = (e: React.DragEvent) => e.dataTransfer.types.includes('Files');

  const handleDragEnter = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setDragActive(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepthRef.current = 0;
    setDragActive(false);
    if (!connected) return;
    void handleFiles(Array.from(e.dataTransfer.files));
  };

  const handleSend = () => {
    if (!wsRef.current?.connected || !canSend) return;

    const content = composeOutgoing(input, readyAttachments);
    if (!content) return;

    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content,
        timestamp: new Date(),
      },
    ]);

    try {
      wsRef.current.sendMessage(content);
      setTyping(true);
      pendingContentRef.current = '';
    } catch {
      setError('Failed to send message. Please try again.');
    }

    // Drop everything that was sent and release its preview URL; keep failed
    // uploads visible so they can be retried or dismissed explicitly.
    setAttachments((prev) => {
      const kept: Attachment[] = [];
      for (const attachment of prev) {
        if (attachment.status === 'error') {
          kept.push(attachment);
        } else if (attachment.previewUrl) {
          URL.revokeObjectURL(attachment.previewUrl);
        }
      }
      return kept;
    });

    setInput('');
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  const handleCopy = useCallback((msgId: string, content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedId(msgId);
      setTimeout(() => setCopiedId((prev) => (prev === msgId ? null : prev)), 2000);
    });
  }, []);

  return (
    <div
      className="relative flex flex-col h-[calc(100vh-3.5rem)]"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {dragActive && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-blue-500 bg-blue-950/60">
          <p className="text-sm font-medium text-blue-200">Drop files to upload</p>
        </div>
      )}

      {/* Connection status bar */}
      {error && (
        <div className="px-4 py-2 bg-red-900/30 border-b border-red-700 flex items-center gap-2 text-sm text-red-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Bot className="h-12 w-12 mb-3 text-gray-600" />
            <p className="text-lg font-medium">ZeroClaw Agent</p>
            <p className="text-sm mt-1">Send a message to start the conversation</p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`group flex items-start gap-3 ${
              msg.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                msg.role === 'user'
                  ? 'bg-blue-600'
                  : 'bg-gray-700'
              }`}
            >
              {msg.role === 'user' ? (
                <User className="h-4 w-4 text-white" />
              ) : (
                <Bot className="h-4 w-4 text-white" />
              )}
            </div>
            <div className="relative max-w-[75%]">
              <div
                className={`rounded-xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-100 border border-gray-700'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap break-words">{msg.content}</p>
                <p
                  className={`text-xs mt-1 ${
                    msg.role === 'user' ? 'text-blue-200' : 'text-gray-500'
                  }`}
                >
                  {msg.timestamp.toLocaleTimeString()}
                </p>
              </div>
              <button
                onClick={() => handleCopy(msg.id, msg.content)}
                aria-label="Copy message"
                className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-400 hover:text-white"
              >
                {copiedId === msg.id ? (
                  <Check className="h-3.5 w-3.5 text-green-400" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          </div>
        ))}

        {typing && (
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
              <Bot className="h-4 w-4 text-white" />
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-3">
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <p className="text-xs text-gray-500 mt-1">Typing...</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-800 bg-gray-900 p-4">
        <div className="max-w-4xl mx-auto space-y-2">
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {attachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className={`flex items-center gap-2 rounded-lg border px-2 py-1.5 ${
                    attachment.status === 'error'
                      ? 'border-red-700 bg-red-900/30'
                      : 'border-gray-700 bg-gray-800'
                  }`}
                >
                  {attachment.previewUrl ? (
                    <img
                      src={attachment.previewUrl}
                      alt={attachment.name}
                      className="h-8 w-8 flex-shrink-0 rounded object-cover"
                    />
                  ) : (
                    <FileText className="h-4 w-4 flex-shrink-0 text-gray-400" />
                  )}
                  <div className="min-w-0">
                    <p className="max-w-[12rem] truncate text-xs text-gray-200">
                      {attachment.name}
                    </p>
                    <p
                      className={`text-[11px] ${
                        attachment.status === 'error'
                          ? 'text-red-300'
                          : 'text-gray-500'
                      }`}
                    >
                      {attachment.status === 'uploading' && 'Uploading...'}
                      {attachment.status === 'ready' && formatBytes(attachment.size)}
                      {attachment.status === 'error' &&
                        (attachment.error ?? 'Upload failed')}
                    </p>
                  </div>
                  {attachment.status === 'uploading' && (
                    <Loader2 className="h-3.5 w-3.5 flex-shrink-0 animate-spin text-gray-400" />
                  )}
                  <button
                    onClick={() => removeAttachment(attachment.id)}
                    aria-label={`Remove ${attachment.name}`}
                    className="flex-shrink-0 rounded p-0.5 text-gray-500 hover:bg-gray-700 hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {hasNonImageAttachment && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-700 bg-amber-900/20 px-3 py-2 text-xs text-amber-300">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
              <span>
                This chat sends a single turn with no tools, so the agent cannot open
                non-image files. Only the stored path is included as text — use a
                tool-enabled channel if the agent needs to read the contents.
              </span>
            </div>
          )}

          <div className="flex items-end gap-3">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleFileInputChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!connected}
              aria-label="Attach files"
              title="Attach files — images are sent to the model, other files are only uploaded"
              className="flex-shrink-0 rounded-xl border border-gray-700 bg-gray-800 p-3 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white disabled:opacity-50"
            >
              <Paperclip className="h-5 w-5" />
            </button>
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder={
                  connected
                    ? 'Type a message, or paste / drop a file...'
                    : 'Connecting...'
                }
                disabled={!connected}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 resize-none overflow-y-auto"
                style={{ minHeight: '44px', maxHeight: '200px' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!connected || !canSend}
              className="flex-shrink-0 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-xl p-3 transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
        <div className="flex items-center justify-center mt-2 gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              connected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-xs text-gray-500">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        {attachments.length > 0 && (
          <p className="mt-1 text-center text-[11px] text-gray-600">
            Images are inlined for vision-capable providers only (defaults: 4 images,
            5 MB each — see the [multimodal] config section).
          </p>
        )}
      </div>
    </div>
  );
}
