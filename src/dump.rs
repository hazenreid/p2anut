// 위키 덤프 본문 정리 순서:
//   1) XML 엔티티 해제 (덤프 XML이 위키 원문을 한 번 더 이스케이프함: 원문 `<ref>` → `&lt;ref&gt;`, 원문 `&nbsp;` → `&amp;nbsp;`)
//   2) 위키 마크업 제거
//   3) 위키 원문 자체의 HTML 엔티티 해제 (`&nbsp;`, `&ndash;` 등). 모르는 엔티티는 공백으로 바꿔 글자로 세지 않음
// 3)을 2) 뒤에 해야 원문의 `&lt;`(글자 그대로의 <)가 태그로 오인되지 않음

// 문서 끝의 정형 섹션 제목 (내용이 대부분 인용·링크 목록이고, 제목 단어도 모든 문서에서 반복됨)
// 이 중 하나가 2단계 제목(== X ==)으로 처음 나오는 곳부터 문서 끝까지 버림
const TAIL_SECTIONS: [&[u8]; 17] = [
  b"references", b"reference", b"notes", b"note", b"footnotes", b"citations", b"sources",
  b"see also", b"external links", b"external link", b"further reading", b"bibliography",
  b"notes and references", b"references and notes", b"works cited", b"sources and further reading",
  b"notes and sources",
];

// 끝 섹션이 시작되는 줄의 위치 (마크업 제거 전, XML 엔티티만 푼 원문 기준)
pub fn tail_section_start(text: &[u8]) -> Option<usize> {
  let mut start = 0;
  for line in text.split(|&b| b == b'\n') {
    let t = line.trim_ascii();
    if t.len() > 4 && t.starts_with(b"==") && t.ends_with(b"==") && t[2] != b'=' && t[t.len() - 3] != b'=' {
      let title = t[2..t.len() - 2].trim_ascii();
      if TAIL_SECTIONS.iter().any(|s| s.eq_ignore_ascii_case(title)) {
        return Some(start);
      }
    }
    start += line.len() + 1;
  }
  None
}

// 위키 본문 텍스트 정리기 (버퍼 재사용)
pub struct ArticleCleaner {
  decoded: Vec<u8>,
  stripped: Vec<u8>,
  text: Vec<u8>,
  // 마지막 clean()에서 끝 섹션을 잘라 냈는지 (검증 도구용)
  pub tail_cut: bool,
}

impl ArticleCleaner {
  pub fn new() -> Self {
    Self {
      decoded: Vec::with_capacity(128 * 1024),
      stripped: Vec::with_capacity(128 * 1024),
      text: Vec::with_capacity(128 * 1024),
      tail_cut: false,
    }
  }

  // raw: 덤프 XML의 <text> 내용 (XML 이스케이프된 상태 그대로)
  pub fn clean(&mut self, raw: &[u8]) -> &[u8] {
    decode_html_entities_into(raw, &mut self.decoded);
    let tail = tail_section_start(&self.decoded);
    self.tail_cut = tail.is_some();
    if let Some(p) = tail { self.decoded.truncate(p); }
    strip_wiki_markup_into(&self.decoded, &mut self.stripped);
    decode_wiki_entities_into(&self.stripped, &mut self.text);
    &self.text
  }
}

// 넘겨주기 문서인지 (#REDIRECT, #Redirect 등 대소문자 무관, 앞 공백 허용)
pub fn is_redirect(raw: &[u8]) -> bool {
  let t = raw.trim_ascii_start();
  t.len() >= 9 && t[..9].eq_ignore_ascii_case(b"#redirect")
}

// XML 단계 엔티티 해제. 모르는 엔티티는 글자 그대로 둠
pub fn decode_html_entities_into(input: &[u8], out: &mut Vec<u8>) {
  decode_entities_into(input, out, false);
}

// 위키 원문 단계 엔티티 해제. 모르는 이름 엔티티(&ndash; 등)는 공백으로 바꿔 연속을 끊음
pub fn decode_wiki_entities_into(input: &[u8], out: &mut Vec<u8>) {
  decode_entities_into(input, out, true);
}

fn decode_entities_into(input: &[u8], out: &mut Vec<u8>, unknown_as_space: bool) {
  out.clear();
  let mut i = 0;
  while i < input.len() {
    if input[i] != b'&' { out.push(input[i]); i += 1; continue; }
    let rest = &input[i + 1..];
    if let Some(end) = rest.iter().take(12).position(|&b| b == b';') {
      let entity = &rest[..end];
      match entity {
        b"lt"   => { out.push(b'<'); i += end + 2; continue; }
        b"gt"   => { out.push(b'>'); i += end + 2; continue; }
        b"amp"  => { out.push(b'&'); i += end + 2; continue; }
        b"quot" => { out.push(b'"'); i += end + 2; continue; }
        b"apos" | b"#39" => { out.push(b'\''); i += end + 2; continue; }
        b"nbsp" => { out.push(b' '); i += end + 2; continue; }
        _ if entity.starts_with(b"#") => {
          let code_bytes = &entity[1..];
          let code: Option<u32> = if code_bytes.starts_with(b"x") || code_bytes.starts_with(b"X") {
            u32::from_str_radix(std::str::from_utf8(&code_bytes[1..]).unwrap_or(""), 16).ok()
          } else {
            std::str::from_utf8(code_bytes).ok().and_then(|s| s.parse().ok())
          };
          if let Some(ch) = code.and_then(char::from_u32) {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            i += end + 2;
            continue;
          }
        }
        _ if unknown_as_space
          && entity.first().is_some_and(u8::is_ascii_alphabetic)
          && entity.iter().all(u8::is_ascii_alphanumeric) => {
          out.push(b' ');
          i += end + 2;
          continue;
        }
        _ => {}
      }
    }
    out.push(b'&');
    i += 1;
  }
}

// rest가 "[["로 시작할 때 짝이 맞는 "]]"의 위치 (중첩 [[ ]] 고려)
fn matching_link_end(rest: &[u8]) -> Option<usize> {
  let mut depth = 0usize;
  let mut j = 0;
  while j + 1 < rest.len() {
    if rest[j] == b'[' && rest[j + 1] == b'[' { depth += 1; j += 2; }
    else if rest[j] == b']' && rest[j + 1] == b']' {
      depth -= 1;
      if depth == 0 { return Some(j); }
      j += 2;
    } else { j += 1; }
  }
  None
}

// 화면에 글자가 보이지 않는 링크인지: 파일·분류 링크, 그리고 언어 간 링크([[fr:Paris]])
// 그 밖의 콜론 링크([[Star Trek: The Next Generation]], [[wikt:hello|hello]])는 보이는 글자를 남김
fn is_hidden_link(target: &[u8], has_pipe: bool) -> bool {
  let Some(colon) = target.iter().position(|&b| b == b':') else { return false; };
  let prefix = target[..colon].trim_ascii();
  const HIDDEN_NS: [&[u8]; 4] = [b"file", b"image", b"media", b"category"];
  if HIDDEN_NS.iter().any(|ns| ns.eq_ignore_ascii_case(prefix)) { return true; }
  // 언어 코드: 소문자 2~3자, 뒤에 "-yue" 같은 소문자 꼬리가 붙을 수 있음 (zh-yue, be-tarask)
  let lower = |p: &[u8]| !p.is_empty() && p.iter().all(u8::is_ascii_lowercase);
  let mut parts = prefix.split(|&b| b == b'-');
  !has_pipe
    && parts.next().is_some_and(|p| (2..=3).contains(&p.len()) && lower(p))
    && parts.all(lower)
}

// 내용까지 통째로 지우는 태그 (각주, 수식, 코드, 화학식, 갤러리 등 본문 문장이 아닌 것)
const DROP_TAGS: [&[u8]; 18] = [
  b"ref", b"references", b"math", b"chem", b"ce", b"syntaxhighlight", b"source", b"pre", b"code",
  b"gallery", b"timeline", b"score", b"hiero", b"graph", b"mapframe", b"maplink", b"templatedata", b"imagemap",
];

// "<name" 다음 글자가 태그 이름의 끝인지
fn is_tag_name_end(b: Option<&u8>) -> bool {
  b.is_none_or(|&b| b == b'>' || b == b'/' || b.is_ascii_whitespace())
}

// hay에서 닫는 태그 </name ...> 를 찾아 그 '>' 바로 뒤 위치를 돌려줌 (대소문자 무관)
fn find_closing_tag(hay: &[u8], name: &[u8]) -> Option<usize> {
  let mut j = 0;
  while j + 2 + name.len() <= hay.len() {
    if hay[j] == b'<' && hay[j + 1] == b'/'
      && hay[j + 2..j + 2 + name.len()].eq_ignore_ascii_case(name)
      && is_tag_name_end(hay.get(j + 2 + name.len())) {
      let gt = hay[j..].iter().position(|&b| b == b'>')?;
      return Some(j + gt + 1);
    }
    j += 1;
  }
  None
}

pub fn strip_wiki_markup_into(input: &[u8], out: &mut Vec<u8>) {
  out.clear();
  let mut i = 0;
  let mut tmpl_depth = 0i32;
  let mut table_depth = 0i32;

  while i < input.len() {
    let rest = &input[i..];

    // 템플릿을 표보다 먼저 처리: 표 안의 {{x|}} 에 들어 있는 "|}"를 표 끝으로 오인하지 않도록
    if rest.starts_with(b"{{") { tmpl_depth += 1; i += 2; continue; }
    if tmpl_depth > 0 {
      if rest.starts_with(b"}}") { tmpl_depth -= 1; i += 2; }
      else { i += 1; }
      continue;
    }

    // 위키 표 {| ... |} 는 셀 구분자·속성이 대부분이라 통째로 제외 (중첩 표 포함)
    if rest.starts_with(b"{|") { table_depth += 1; i += 2; continue; }
    if table_depth > 0 {
      if rest.starts_with(b"|}") { table_depth -= 1; i += 2; }
      else { i += 1; }
      continue;
    }

    // 굵게/기울임 ('' ''' ''''') 은 지워서 앞뒤 글자가 자연스럽게 이어지게 함
    if rest.starts_with(b"''") {
      i += rest.iter().take_while(|&&b| b == b'\'').count();
      continue;
    }

    // 매직 워드 (__NOTOC__, __TOC__ 등)
    if rest.starts_with(b"__") {
      let n = rest[2..].iter().take_while(|b| b.is_ascii_uppercase()).count();
      if n >= 2 && rest[2 + n..].starts_with(b"__") {
        i += n + 4;
        continue;
      }
    }

    if rest.starts_with(b"<!--") {
      i += rest.windows(3).position(|w| w == b"-->").map_or(rest.len(), |p| p + 3);
      continue;
    }

    if rest.starts_with(b"<") {
      let name_len = rest[1..].iter().take_while(|b| b.is_ascii_alphanumeric()).count();
      let name = &rest[1..1 + name_len];

      // 내용까지 지우는 태그: <math>...</math>, <ref name=x/>, <ref>...</ref> 등
      if name_len > 0 && is_tag_name_end(rest.get(1 + name_len))
        && DROP_TAGS.iter().any(|t| t.eq_ignore_ascii_case(name)) {
        let Some(gt) = rest.iter().position(|&b| b == b'>') else { i += 1 + name_len; continue; };
        if rest[gt - 1] == b'/' {
          i += gt + 1;
        } else if let Some(p) = find_closing_tag(&rest[gt + 1..], name) {
          i += gt + 1 + p;
        } else {
          i += gt + 1;
        }
        continue;
      }

      // 그 밖의 태그는 태그 자체만 지우고 내용은 남김 (<br>, </div>, <span ...>, <nowiki> 등)
      // "<" 뒤가 영문자·'/'·'!'이고 같은 줄에서 '>'로 닫힐 때만 태그로 봄 ("x < 5" 같은 부등호는 글자 그대로)
      if rest.get(1).is_some_and(|&b| b.is_ascii_alphabetic() || b == b'/' || b == b'!')
        && let Some(gt) = rest[1..].iter().position(|&b| matches!(b, b'>' | b'<' | b'\n')).filter(|&p| rest[1 + p] == b'>') {
        i += gt + 2;
        continue;
      }
    }

    if rest.starts_with(b"[[") {
      if let Some(end) = matching_link_end(rest) {
        let content = &rest[2..end];
        // [[:Category:X]] 처럼 앞에 콜론이 붙으면 분류·파일이 아니라 보이는 링크
        let (forced_visible, content) = match content.strip_prefix(b":") {
          Some(c) => (true, c),
          None => (false, content),
        };
        let pipe = content.iter().position(|&b| b == b'|');
        let target = &content[..pipe.unwrap_or(content.len())];
        if !forced_visible && is_hidden_link(target, pipe.is_some()) {
          // 파일·분류·언어 간 링크 — 캡션 안의 중첩 링크까지 통째로 제외
        } else {
          let shown = pipe.map_or(content, |p| &content[p + 1..]);
          // 표시 텍스트 안에 다시 마크업(중첩 링크, 굵게 등)이 있을 때만 재귀 처리
          if shown.iter().any(|&b| matches!(b, b'[' | b'\'' | b'{' | b'<' | b'_')) {
            let mut inner = Vec::new();
            strip_wiki_markup_into(shown, &mut inner);
            out.extend_from_slice(&inner);
          } else {
            out.extend_from_slice(shown);
          }
        }
        i += end + 2;
      } else {
        i += 2;
      }
      continue;
    }

    if rest.starts_with(b"[http") || rest.starts_with(b"[//") {
      i += rest.iter().position(|&b| b == b']').map_or(1, |p| p + 1);
      continue;
    }

    out.push(input[i]);
    i += 1;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn strip(s: &str) -> String {
    let mut out = Vec::new();
    strip_wiki_markup_into(s.as_bytes(), &mut out);
    String::from_utf8(out).unwrap()
  }

  // 덤프 XML 원문(이스케이프된 상태)을 전체 파이프라인으로 정리
  fn clean(xml: &str) -> String {
    let mut c = ArticleCleaner::new();
    String::from_utf8(c.clean(xml.as_bytes()).to_vec()).unwrap()
  }

  #[test]
  fn ref_tags_are_removed_without_eating_text() {
    // 뒤쪽 self-closing ref 때문에 사이 본문이 사라지면 안 됨
    assert_eq!(strip("aa<ref>cite</ref>bb cc<ref name=\"x\"/>dd"), "aabb ccdd");
    assert_eq!(strip("x<ref name=a>c</ref>y"), "xy");
    assert_eq!(strip("p<references/>q"), "pq");
    assert_eq!(strip("p<references>\n<ref name=a>c</ref>\n</references>q"), "pq");
    assert_eq!(strip("x<REF>c</REF>y"), "xy");
  }

  #[test]
  fn templates_comments_and_links() {
    assert_eq!(strip("a{{cite|x={{y}}}}b"), "ab");
    assert_eq!(strip("a<!-- hidden -->b"), "ab");
    assert_eq!(strip("[[Target|shown]] [[plain]] [[File:x.png|cap]]"), "shown plain ");
    assert_eq!(strip("see [http://x.org label] now"), "see  now");
  }

  #[test]
  fn tables_are_removed() {
    assert_eq!(strip("a\n{| class=wikitable\n|-\n| x || y\n|}\nb"), "a\n\nb");
    // 중첩 표, 그리고 표 안 템플릿의 "|}}" 를 표 끝으로 오인하지 않아야 함
    assert_eq!(strip("a{| {{t|}} {| inner |} still |}b"), "ab");
  }

  #[test]
  fn nested_links_and_bold() {
    assert_eq!(strip("x [[File:a.png|thumb|cap with [[link]] in it]] y"), "x  y");
    assert_eq!(strip("[[a|b [[c|e]] d]]"), "b e d");
    assert_eq!(strip("[[Paris|'''Paris''']] is"), "Paris is");
    assert_eq!(strip("'''Paris''' is ''very'' old"), "Paris is very old");
    assert_eq!(strip("don't"), "don't");
  }

  #[test]
  fn html_entities() {
    let mut out = Vec::new();
    decode_html_entities_into(b"a&lt;b&gt;&amp;&#65;&#x42;&nbsp;&bogus;", &mut out);
    assert_eq!(out, b"a<b>&AB &bogus;");
    // 위키 원문 단계에서는 모르는 이름 엔티티를 공백으로
    decode_wiki_entities_into(b"1990&ndash;2000 &bogus; AT&T &#;", &mut out);
    assert_eq!(out, b"1990 2000   AT&T &#;");
  }

  #[test]
  fn wiki_source_entities_are_not_counted_as_letters() {
    // 원문 &nbsp; 는 덤프에서 &amp;nbsp; 로 이중 이스케이프됨
    assert_eq!(clean("10&amp;nbsp;km, 1990&amp;ndash;2000 &amp;minus;5"), "10 km, 1990 2000  5");
    // 원문의 &lt; 는 글자 그대로의 < 이므로 태그로 오인해 뒤 본문을 지우면 안 됨
    assert_eq!(clean("a &amp;lt;b and c&amp;gt; d"), "a <b and c> d");
  }

  #[test]
  fn content_tags_are_dropped() {
    assert_eq!(clean("where &lt;math&gt;\\frac{a}{b}&lt;/math&gt; holds"), "where  holds");
    assert_eq!(strip("use <syntaxhighlight lang=\"python\">print(value)</syntaxhighlight> here"), "use  here");
    assert_eq!(strip("a <code>foo_bar</code> b <pre>x</pre> c"), "a  b  c");
    assert_eq!(strip("<gallery>\nFile:A.jpg|Caption one\n</gallery> after"), " after");
    assert_eq!(strip("H<chem>H2O</chem> <ce>NaCl</ce>."), "H .");
    assert_eq!(strip("x<math display=block/>y"), "xy");
    // 이름이 비슷한 다른 태그(<code>와 <codex>, <ref>와 <refx>)는 내용을 남김
    assert_eq!(strip("<codex>keep</codex> <refx>keep</refx>"), "keep keep");
  }

  #[test]
  fn less_than_sign_is_not_a_tag() {
    assert_eq!(strip("if x < 5 then the result is large <br> next"), "if x < 5 then the result is large  next");
    assert_eq!(strip("a <b>bold</b> c"), "a bold c");
    assert_eq!(strip("x <y\nz> w"), "x <y\nz> w");
    assert_eq!(strip("a<nowiki/>b <nowiki>kept</nowiki>"), "ab kept");
  }

  #[test]
  fn colon_links_keep_visible_text() {
    assert_eq!(strip("[[Star Trek: The Next Generation]] aired"), "Star Trek: The Next Generation aired");
    assert_eq!(strip("[[wikt:hello|hello]] there"), "hello there");
    assert_eq!(strip("[[:Category:Foo|foo list]]"), "foo list");
    assert_eq!(strip("a[[Category:People]]b[[image:x.png]]c[[ fr :Paris]]d[[zh-yue:X]]e"), "abcde");
    assert_eq!(strip("[[Media:Sound.ogg|listen]]."), ".");
  }

  #[test]
  fn tail_sections_are_cut() {
    let body = "Intro text.\n== History ==\nOld story.\n=== Notes ===\nsub note kept\n";
    let tail = "==See also==\n* [[Other]]\n== References ==\n{{reflist}}\n[[Category:X]]";
    assert_eq!(tail_section_start(format!("{body}{tail}").as_bytes()), Some(body.len()));
    // 대소문자·공백 변형
    assert_eq!(tail_section_start(b"a\n  ==  EXTERNAL LINKS  ==  \nb"), Some(2));
    // 3단계 제목, 본문 섹션, 제목이 아닌 줄은 자르지 않음
    assert_eq!(tail_section_start(b"a\n=== References ===\nb"), None);
    assert_eq!(tail_section_start(b"a\n== Referenced works ==\nb"), None);
    assert_eq!(tail_section_start(b"see also == references == here"), None);

    let mut c = ArticleCleaner::new();
    let out = String::from_utf8(c.clean(format!("{body}{tail}").as_bytes()).to_vec()).unwrap();
    assert!(c.tail_cut);
    assert!(out.contains("sub note kept") && !out.contains("Other") && !out.contains("References"), "{out}");
    c.clean(b"no tail here");
    assert!(!c.tail_cut);
  }

  #[test]
  fn magic_words_and_redirects() {
    assert_eq!(strip("__NOTOC__ Text __TOC__ snake_case __x__"), " Text  snake_case __x__");
    assert!(is_redirect(b"#REDIRECT [[Foo]]"));
    assert!(is_redirect(b"#Redirect [[Foo]]"));
    assert!(is_redirect(b"  #redirect[[Foo]]"));
    assert!(!is_redirect(b"#Redir"));
    assert!(!is_redirect(b"Normal text"));
  }
}
