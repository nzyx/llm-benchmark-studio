/**
 * Minimal inline formatter for registry text.
 *
 * The parameter registry stores explanations as plain strings with a couple
 * of conventions (`**bold**`, `` `code` ``, blank line = paragraph).  We
 * escape first, then apply the markup, so nothing from the payload can ever
 * be interpreted as HTML.
 */

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function inline(input: string): string {
  return escapeHtml(input)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>')
}

export function InlineText({ text }: { text: string }) {
  return <span dangerouslySetInnerHTML={{ __html: inline(text) }} />
}

export default function RichText({ text }: { text: string }) {
  const blocks = (text || '').split(/\n{2,}/).filter(Boolean)
  return (
    <>
      {blocks.map((block, i) => (
        <p key={i} dangerouslySetInnerHTML={{ __html: inline(block) }} />
      ))}
    </>
  )
}
